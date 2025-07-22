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
from typing import Optional, Dict
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import speech_pb2
import speech_pb2_grpc
from functools import lru_cache

# 环境配置
os.environ['MODELSCOPE_CACHE'] = './model_cache'
os.environ['MODEL_SCOPE_NO_AST_SCAN'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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
ASR_MODEL = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
PUNC_MODEL = 'damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
TARGET_SAMPLE_RATE = 16000
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB


class AudioProcessor:
    """音频处理工具类"""

    def __init__(self):
        self._resamplers = {}  # 多采样率支持
        self._resampler_lock = threading.Lock()

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio

        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=target_sr
            )

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        return self._resamplers[orig_sr](audio_tensor).squeeze().numpy()

    @staticmethod
    def bytes_to_audio(raw_data: bytes, sample_rate: int) -> np.ndarray:
        """增强版音频数据转换"""
        if not raw_data:
            raise ValueError("空音频数据")

        try:
            # 尝试作为WAV解析
            with io.BytesIO(raw_data) as bio:
                try:
                    audio, _ = sf.read(bio)
                    if len(audio) == 0:
                        raise sf.LibsndfileError("空音频数据")
                    return audio
                except (sf.LibsndfileError, RuntimeError) as e:
                    bio.seek(0)
                    # 作为原始PCM处理
                    if len(raw_data) % 2 != 0:
                        raw_data = raw_data[:-1]  # 确保长度是2的倍数

                    # 安全检查
                    if len(raw_data) < 2:
                        raise ValueError("音频数据过短")

                    audio = np.frombuffer(raw_data, dtype=np.int16)
                    if len(audio) == 0:
                        raise ValueError("PCM数据长度为0")

                    # 安全归一化
                    max_val = np.max(np.abs(audio))
                    if max_val == 0:
                        return np.zeros_like(audio, dtype=np.float32)

                    return audio.astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"音频转换失败: {str(e)}")
            raise ValueError(f"无法解析音频数据: {str(e)}")

class SpeechServicer(speech_pb2_grpc.SpeechServicer):
    def __init__(self):
        self._asr_pipeline = None
        self._punc_pipeline = None
        self._model_lock = threading.Lock()
        self._audio_processor = AudioProcessor()
        self._load_models()

    def _load_models(self):
        """加载ASR和标点模型"""
        with self._model_lock:
            # ASR模型
            if self._asr_pipeline is None:
                self._asr_pipeline = pipeline(
                    task=Tasks.auto_speech_recognition,
                    model=ASR_MODEL,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )

            # 标点模型（延迟加载）
            self._punc_pipeline = None

    def _get_punctuation_model(self):
        """按需加载标点模型"""
        if self._punc_pipeline is None:
            with self._model_lock:
                if self._punc_pipeline is None:
                    self._punc_pipeline = pipeline(
                        task=Tasks.punctuation,
                        model=PUNC_MODEL,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
        return self._punc_pipeline

    @lru_cache(maxsize=100)
    def _add_punctuation(self, text: str) -> str:
        """带缓存的标点添加"""
        if not text.strip():
            return text

        try:
            punc_model = self._get_punctuation_model()
            result = punc_model(text)
            return result[0]['text'] if isinstance(result, list) else text + "。"
        except Exception as e:
            logger.warning(f"标点预测失败: {str(e)}")
            return text + "。"  # 降级处理

    def StreamingRecognize(self, request_iterator, context):
        audio_buffer = bytearray()  # 改为字节缓冲
        sample_rate = None
        is_first_chunk = True

        try:
            for request in request_iterator:
                if sample_rate is None:
                    sample_rate = request.sample_rate
                    logger.info(f"开始流式识别, 采样率: {sample_rate}Hz")

                # 累积原始字节数据
                audio_buffer.extend(request.audio_data)

                # 尝试解析（仅在收到足够数据时）
                if len(audio_buffer) > 1024:  # 至少1KB数据才尝试解析
                    try:
                        # 尝试作为完整WAV解析
                        with io.BytesIO(audio_buffer) as bio:
                            try:
                                audio, _ = sf.read(bio)
                            except sf.LibsndfileError:
                                # 如果不是完整WAV，尝试作为原始PCM处理
                                bio.seek(0)
                                audio = np.frombuffer(bio.read(), dtype=np.int16)
                                audio = audio.astype(np.float32) / 32768.0

                        if audio.ndim > 1:
                            audio = np.mean(audio, axis=1)

                        # 返回临时结果
                        if request.interim_results:
                            resampled_audio = self._audio_processor.resample_audio(
                                audio, sample_rate, TARGET_SAMPLE_RATE
                            )
                            raw_result = self._asr_pipeline(audio_in=resampled_audio)
                            print("raw_result:",raw_result)
                            text = raw_result['text']
                            yield speech_pb2.SpeechResponse(
                                text=text,
                                is_final=False,
                                confidence=0.9
                            )

                    except Exception as e:
                        logger.warning(f"音频块解析失败: {str(e)}")
                        continue

                # 处理最终结果
                if not request.interim_results and len(audio_buffer) > 0:
                    try:
                        with io.BytesIO(audio_buffer) as bio:
                            try:
                                audio, _ = sf.read(bio)
                            except:
                                bio.seek(0)
                                audio = np.frombuffer(bio.read(), dtype=np.int16)
                                audio = audio.astype(np.float32) / 32768.0

                        resampled_audio = self._audio_processor.resample_audio(
                            audio, sample_rate, TARGET_SAMPLE_RATE
                        )
                        text = self._asr_pipeline(audio_in=resampled_audio)['text']
                        yield speech_pb2.SpeechResponse(
                            text=self._add_punctuation(text),
                            is_final=True,
                            confidence=0.95
                        )
                    finally:
                        audio_buffer.clear()


        except Exception as e:
            logger.error(f"流式识别错误: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"处理错误: {str(e)}")


    """非流式识别接口"""

    def Recognize(self, request, context):
        try:
            # 增强参数校验
            if not request.audio_data:
                logger.error("收到空音频数据")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("音频数据不能为空")
                return speech_pb2.SpeechResponse()

            if not hasattr(request, 'sample_rate') or request.sample_rate <= 0:
                logger.error(f"无效采样率: {getattr(request, 'sample_rate', '未设置')}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("采样率必须为正整数")
                return speech_pb2.SpeechResponse()

            # 设置默认采样率（如果客户端未指定）
            sample_rate = request.sample_rate if request.sample_rate > 0 else TARGET_SAMPLE_RATE
            logger.info(f"处理识别请求，采样率: {sample_rate}Hz")

            # 读取音频数据
            try:
                audio = self._audio_processor.bytes_to_audio(
                    request.audio_data,
                    sample_rate
                )
            except ValueError as e:
                logger.error(f"音频数据解析失败: {str(e)}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"无效音频数据: {str(e)}")
                return speech_pb2.SpeechResponse()

            # 检查音频长度
            if len(audio) == 0:
                logger.warning("收到空音频数据")
                return speech_pb2.SpeechResponse(text="", is_final=True)

            # 重采样
            if sample_rate != TARGET_SAMPLE_RATE:
                try:
                    audio = self._audio_processor.resample_audio(
                        audio,
                        sample_rate,
                        TARGET_SAMPLE_RATE
                    )
                except Exception as e:
                    logger.error(f"重采样失败: {str(e)}")
                    raise

            # ASR识别
            try:
                raw_result = self._asr_pipeline(audio_in=audio)
                logger.debug(f"ASR原始结果: {raw_result}")
                text = raw_result.get('text', '')

                return speech_pb2.SpeechResponse(
                    text=self._add_punctuation(text),
                    is_final=True,
                    confidence=raw_result.get('confidence', 0.95)
                )
            except Exception as e:
                logger.error(f"ASR处理失败: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"识别错误: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"处理错误: {str(e)}")
            return speech_pb2.SpeechResponse()


def serve(port=50051):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.max_send_message_length', 50 * 1024 * 1024)
        ]
    )
    speech_pb2_grpc.add_SpeechServicer_to_server(SpeechServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"服务已启动，监听端口 {port}")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()