import os
import io
import threading
import time
import logging
import numpy as np
import torch
import torchaudio
import grpc
import soundfile as sf
from concurrent import futures
from typing import Optional, Tuple, List, Dict
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import speech_pb2
import speech_pb2_grpc

# 环境配置
os.environ['MODELSCOPE_CACHE'] = './model_cache'
os.environ['MODEL_SCOPE_NO_AST_SCAN'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('speech_server.log')
    ]
)
logger = logging.getLogger(__name__)

# 常量配置
MODEL_NAME = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
MODEL_REVISION = 'v1.2.1'
TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 0.1  # 最小音频时长(秒)
MAX_AUDIO_DURATION = 30.0  # 最大音频时长(秒)
DEFAULT_CHUNK_DURATION_MS = 200  # 默认分块时长(毫秒)
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
MAX_QPS = 10  # 最大每秒查询数


class AudioProcessor:
    """增强版音频处理工具类"""

    def __init__(self):
        self._resampler_cache = {}
        self._chunk_size_cache = {}
        self._lock = threading.Lock()

    def calculate_chunk_size(self, sample_rate: int, duration_ms: int = DEFAULT_CHUNK_DURATION_MS) -> int:
        """计算指定时长的音频块大小"""
        key = (sample_rate, duration_ms)
        if key not in self._chunk_size_cache:
            chunk_size = (sample_rate * duration_ms) // 1000
            self._chunk_size_cache[key] = chunk_size
        return self._chunk_size_cache[key]

    def resample_audio(self, audio_np: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        """线程安全的音频重采样"""
        if orig_rate == target_rate:
            return audio_np

        resampler_key = (orig_rate, target_rate)
        if resampler_key not in self._resampler_cache:
            with self._lock:
                if resampler_key not in self._resampler_cache:  # 双重检查锁定
                    self._resampler_cache[resampler_key] = torchaudio.transforms.Resample(
                        orig_freq=orig_rate,
                        new_freq=target_rate
                    )

        audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
        resampled = self._resampler_cache[resampler_key](audio_tensor)
        return resampled.squeeze().numpy()

    def validate_audio_input(self, audio_data: bytes, sample_rate: int) -> Tuple[np.ndarray, int]:
        """支持多种音频格式和数据类型"""
        try:
            with io.BytesIO(audio_data) as bio:
                try:
                    # 尝试解析WAV格式
                    audio_np, sr = sf.read(bio)
                    if audio_np.ndim > 1:  # 多声道转单声道
                        audio_np = np.mean(audio_np, axis=1)
                except sf.LibsndfileError:
                    # 尝试解析原始字节数据
                    try:
                        # 先尝试int16
                        audio_np = np.frombuffer(audio_data, dtype=np.int16)
                        sr = sample_rate
                    except:
                        # 再尝试float32
                        audio_np = np.frombuffer(audio_data, dtype=np.float32)
                        sr = sample_rate
        except Exception as e:
            logger.error(f"音频解析失败: {str(e)}")
            raise ValueError(f"无效的音频格式: {str(e)}")

        # 统一转换为float32并归一化
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
            if np.issubdtype(audio_np.dtype, np.integer):
                audio_np /= np.iinfo(audio_np.dtype).max

        # 检查采样率
        if sr <= 0:
            logger.warning(f"无效采样率: {sr}, 使用默认 {TARGET_SAMPLE_RATE}Hz")
            sr = TARGET_SAMPLE_RATE

        # 检查音频时长
        duration = len(audio_np) / sr
        if duration < MIN_AUDIO_DURATION:
            raise ValueError(f"音频过短: {duration:.2f}s < {MIN_AUDIO_DURATION}s")
        if duration > MAX_AUDIO_DURATION:
            logger.warning(f"长音频将被分段处理: {duration:.2f}s > {MAX_AUDIO_DURATION}s")

        return audio_np, sr

class SpeechServicer(speech_pb2_grpc.SpeechServicer):
    """优化后的语音识别服务实现"""

    def __init__(self):
        self._model_loaded = False
        self._model_lock = threading.Lock()
        self._audio_processor = AudioProcessor()
        self.inference_pipeline = None
        self._performance_stats = {
            'total_requests': 0,
            'success_requests': 0,
            'total_audio_seconds': 0.0,
            'total_processing_time': 0.0,
            'last_reset_time': time.time(),
            'latencies': [],
            'last_request_time': 0
        }
        self._model_last_reload_time = 0
        self._model_reload_interval = 3600  # 1小时重载一次

        # 预加载模型
        self._load_model()

    def _load_model(self):
        """线程安全的模型加载方法"""
        if self._model_loaded and time.time() - self._model_last_reload_time < self._model_reload_interval:
            return

        with self._model_lock:
            if self._model_loaded and time.time() - self._model_last_reload_time < self._model_reload_interval:
                return

            logger.info("加载/重载Paraformer模型...")
            try:
                # 模型配置
                model_config = {
                    'model_revision': MODEL_REVISION,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'ngpu': 1 if torch.cuda.is_available() else 0,
                    'beam_size': 5,
                    'max_length': 50,
                    'quantize': True,
                    'use_onnx': False
                }

                self.inference_pipeline = pipeline(
                    task=Tasks.auto_speech_recognition,
                    model=MODEL_NAME,
                    **model_config
                )
                self._model_loaded = True
                self._model_last_reload_time = time.time()
                logger.info("模型加载成功")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}", exc_info=True)
                self._model_loaded = False
                raise RuntimeError(f"模型加载失败: {str(e)}")

    def _safe_recognize(self, audio_np: np.ndarray, sample_rate: int) -> str:
        """带重试机制的识别方法"""
        for attempt in range(3):  # 最大重试3次
            try:
                if not self._model_loaded or time.time() - self._model_last_reload_time > self._model_reload_interval:
                    self._load_model()

                # 重采样到目标采样率
                if sample_rate != TARGET_SAMPLE_RATE:
                    audio_np = self._audio_processor.resample_audio(audio_np, sample_rate, TARGET_SAMPLE_RATE)

                # 音量检查
                if np.max(np.abs(audio_np)) < 0.01:
                    logger.warning("低音量音频可能影响识别结果")

                # 调用模型
                result = self.inference_pipeline(audio_in=audio_np)

                # 处理模型输出
                if isinstance(result, dict):
                    text = result.get('text', '').strip()
                elif isinstance(result, list):
                    text = ' '.join([r.get('text', '').strip() for r in result if isinstance(r, dict)])
                else:
                    text = ""

                if not text:
                    logger.warning("模型返回空结果")
                else:
                    logger.info(f"识别结果: {text}")

                return text
            except Exception as e:
                logger.error(f"识别失败（尝试 {attempt + 1}/3）: {str(e)}")
                self._model_loaded = False
                if attempt == 2:  # 最后一次尝试仍然失败
                    raise
                time.sleep(1)  # 等待后重试

    def _check_request_limit(self, context) -> bool:
        """检查请求频率限制"""
        now = time.time()
        with self._model_lock:
            # QPS检查
            if self._performance_stats['total_requests'] / max(1, now - self._performance_stats[
                'last_reset_time']) > MAX_QPS:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("请求过于频繁")
                return False
            self._performance_stats['last_request_time'] = now
            return True

    def Recognize(self, request, context):
        """处理单次语音识别请求"""
        start_time = time.time()
        self._performance_stats['total_requests'] += 1

        # 请求限制检查
        if not self._check_request_limit(context):
            return speech_pb2.SpeechResponse()

        # 请求大小检查
        if len(request.audio_data) > MAX_REQUEST_SIZE:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"音频数据过大（>{MAX_REQUEST_SIZE // 1024 // 1024}MB）")
            return speech_pb2.SpeechResponse()

        try:
            logger.info(f"收到音频数据, 长度: {len(request.audio_data)} bytes")

            # 验证并预处理音频
            audio_np, sample_rate = self._audio_processor.validate_audio_input(
                request.audio_data,
                request.sample_rate
            )

            # 语音识别
            text = self._safe_recognize(audio_np, sample_rate)

            # 更新性能统计
            audio_duration = len(audio_np) / sample_rate
            processing_time = time.time() - start_time
            self._update_stats(success=True, audio_duration=audio_duration, processing_time=processing_time)

            return speech_pb2.SpeechResponse(
                text=text,
                is_final=True,
                confidence=0.95  # 固定置信度值
            )
        except Exception as e:
            logger.error(f"识别错误: {str(e)}", exc_info=True)
            self._update_stats(success=False)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"识别错误: {str(e)}")
            return speech_pb2.SpeechResponse()

    def StreamingRecognize(self, request_iterator, context):
        """处理流式语音识别请求"""
        start_time = time.time()
        self._performance_stats['total_requests'] += 1

        if not self._check_request_limit(context):
            return

        try:
            audio_buffer = []
            sample_rate = None
            last_process_time = time.time()

            for request in request_iterator:
                # 首次请求初始化采样率
                if not sample_rate:
                    sample_rate = request.sample_rate
                    logger.info(f"流式识别开始, 采样率: {sample_rate}Hz")

                # 检查请求大小
                if len(request.audio_data) > MAX_REQUEST_SIZE:
                    logger.warning("流式请求中单个数据包过大")
                    continue

                # 收集音频数据
                try:
                    audio_np, _ = self._audio_processor.validate_audio_input(
                        request.audio_data,
                        sample_rate
                    )
                    audio_buffer.append(audio_np)
                except ValueError as e:
                    logger.warning(f"无效音频块: {str(e)}")
                    continue

                # 实时处理（每1秒或收到结束标志时识别）
                current_time = time.time()
                if request.interim_results or len(
                        audio_buffer) >= sample_rate or current_time - last_process_time > 1.0:
                    combined_audio = np.concatenate(audio_buffer)
                    try:
                        text = self._safe_recognize(combined_audio, sample_rate)

                        yield speech_pb2.SpeechResponse(
                            text=text,
                            is_final=not request.interim_results,
                            confidence=0.9
                        )

                        if not request.interim_results:
                            audio_buffer.clear()
                        last_process_time = current_time
                    except Exception as e:
                        logger.error(f"流式识别处理失败: {str(e)}")
                        continue

            # 处理剩余音频
            if audio_buffer:
                combined_audio = np.concatenate(audio_buffer)
                try:
                    text = self._safe_recognize(combined_audio, sample_rate)
                    yield speech_pb2.SpeechResponse(
                        text=text,
                        is_final=True,
                        confidence=0.9
                    )
                except Exception as e:
                    logger.error(f"流式收尾识别失败: {str(e)}")

            # 更新性能统计
            audio_duration = sum(len(chunk) for chunk in audio_buffer) / sample_rate
            processing_time = time.time() - start_time
            self._update_stats(success=True, audio_duration=audio_duration, processing_time=processing_time)

        except Exception as e:
            logger.error(f"流式识别错误: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"流式识别错误: {str(e)}")

    def _update_stats(self, success: bool, audio_duration: float = 0.0, processing_time: float = 0.0):
        """更新性能统计"""
        with self._model_lock:
            if success:
                self._performance_stats['success_requests'] += 1
                self._performance_stats['total_audio_seconds'] += audio_duration
                self._performance_stats['total_processing_time'] += processing_time
                self._performance_stats['latencies'].append(processing_time)
                if len(self._performance_stats['latencies']) > 1000:
                    self._performance_stats['latencies'].pop(0)

    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        with self._model_lock:
            stats = self._performance_stats.copy()
            uptime = time.time() - stats['last_reset_time']

            if stats['success_requests'] > 0:
                stats['avg_rtf'] = stats['total_processing_time'] / stats['total_audio_seconds']
                stats['success_rate'] = stats['success_requests'] / stats['total_requests']
                stats['current_qps'] = stats['total_requests'] / uptime

                if stats['latencies']:
                    latencies = sorted(stats['latencies'])
                    stats.update({
                        'p99_latency': latencies[int(len(latencies) * 0.99)],
                        'p95_latency': latencies[int(len(latencies) * 0.95)],
                        'avg_latency': sum(latencies) / len(latencies)
                    })
            else:
                stats.update({
                    'avg_rtf': 0.0,
                    'success_rate': 0.0,
                    'current_qps': 0.0
                })

            stats['uptime'] = uptime
            return stats

    def reset_performance_stats(self):
        """重置性能统计"""
        with self._model_lock:
            self._performance_stats = {
                'total_requests': 0,
                'success_requests': 0,
                'total_audio_seconds': 0.0,
                'total_processing_time': 0.0,
                'last_reset_time': time.time(),
                'latencies': [],
                'last_request_time': 0
            }


def serve(max_workers: int = 10, port: int = 50051, max_msg_size: int = 50):
    """启动gRPC服务器"""
    # 配置gRPC服务器
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_receive_message_length', max_msg_size * 1024 * 1024),
            ('grpc.max_send_message_length', max_msg_size * 1024 * 1024),
            ('grpc.so_reuseport', 1)
        ]
    )

    # 添加服务
    speech_servicer = SpeechServicer()
    speech_pb2_grpc.add_SpeechServicer_to_server(speech_servicer, server)

    # 监听端口
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logger.info(f"语音识别服务已启动, 监听端口 {port}, 最大消息大小 {max_msg_size}MB")

    try:
        # 定期打印性能指标
        while True:
            time.sleep(60)
            metrics = speech_servicer.get_performance_metrics()
            logger.info(
                "性能指标 - "
                f"请求数: {metrics['total_requests']}, "
                f"成功率: {metrics['success_rate']:.2%}, "
                f"QPS: {metrics['current_qps']:.1f}, "
                f"平均RTF: {metrics['avg_rtf']:.2f}, "
                f"P99延迟: {metrics.get('p99_latency', 0):.3f}s, "
                f"运行时间: {metrics['uptime']:.0f}s"
            )

            # 每小时自动重置统计
            if metrics['uptime'] > 3600:
                speech_servicer.reset_performance_stats()
    except KeyboardInterrupt:
        logger.info("正在关闭服务器...")
        server.stop(0).wait()
        logger.info("服务器已关闭")


if __name__ == '__main__':
    serve()