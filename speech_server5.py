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
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import speech_pb2
import speech_pb2_grpc
import webrtcvad

# ---------- 环境与日志 ----------
os.environ['MODELSCOPE_CACHE'] = './model_cache'
os.environ['MODEL_SCOPE_NO_AST_SCAN'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('speech_server.log')]
)
logger = logging.getLogger(__name__)

# ---------- 常量 ----------
ASR_MODEL = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
TARGET_SAMPLE_RATE = 16000
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB


class AudioProcessor:
    def __init__(self):
        self._resamplers = {}
        self._resampler_lock = threading.Lock()

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=target_sr
            )
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        return self._resamplers[orig_sr](audio_tensor).squeeze().numpy()

    @staticmethod
    def bytes_to_audio(raw_data: bytes, sample_rate: int) -> np.ndarray:
        if not raw_data:
            raise ValueError("空音频数据")
        try:
            with io.BytesIO(raw_data) as bio:
                try:
                    audio, _ = sf.read(bio)
                    if len(audio) == 0:
                        raise sf.LibsndfileError("空音频数据")
                    return audio
                except sf.LibsndfileError:
                    bio.seek(0)
                    if len(raw_data) % 2 != 0:
                        raw_data = raw_data[:-1]
                    audio = np.frombuffer(raw_data, dtype=np.int16)
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
        with self._model_lock:
            if self._asr_pipeline is None:
                self._asr_pipeline = pipeline(
                    task=Tasks.auto_speech_recognition,
                    model=ASR_MODEL,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            self._punc_pipeline = None

    # ---------- 流式识别 ----------
    # 立即触发条件（最快0.5 秒）：
    #     当request.interim_results = True（请求中间结果）且缓冲区有 ≥0.5秒 音频时
    #     这是最快获取结果的路径，适合需要实时反馈的场景
    # 常规触发条件（约2秒）：
    #     距离上次识别 ≥2秒（segment_duration）时触发
    #     这是标准情况，平衡了延迟和识别准确性的权衡
    # 保护性触发条件（最长3秒）：
    #     当缓冲区积累到 ≥3秒（segment_duration * 1.5）时强制处理
    #     这是防止缓冲区无限增长的安全机制
    def StreamingRecognize(self, request_iterator, context):
        audio_buffer = bytearray()
        sample_rate = None
        last_recognition_time = time.time()
        segment_duration = 2.0  # 目标分段时长
        vad = webrtcvad.Vad(3)  #  1-3 3高灵敏度

        for request in request_iterator:
            if sample_rate is None:
                sample_rate = request.sample_rate
                # 检查采样率是否被VAD支持
                if sample_rate not in [8000, 16000, 32000, 48000]:
                    logger.warning(f"不支持的采样率: {sample_rate}, 将使用16000Hz进行VAD检测")
                    vad_sample_rate = 16000
                else:
                    vad_sample_rate = sample_rate

            if not request.audio_data:
                # 处理空包逻辑
                pass

            audio_buffer.extend(request.audio_data)
            current_time = time.time()
            buffer_duration = len(audio_buffer) / (sample_rate * 2)  # 假设16位采样

            # 检查分段条件
            should_process = (
                    current_time - last_recognition_time >= segment_duration or
                    buffer_duration >= segment_duration * 1.5 or
                    (request.interim_results and buffer_duration >= 0.5)
            )

            if should_process:
                # 计算VAD帧大小(30ms)
                frame_size = int(0.03 * vad_sample_rate) * 2  # 16bit = 2字节

                # 确保有足够数据
                if len(audio_buffer) >= frame_size:
                    best_split = len(audio_buffer)

                    # 从后向前找最近的静音点
                    for i in range(len(audio_buffer) - frame_size,
                                   max(0, len(audio_buffer) - frame_size * 10),
                                   -frame_size):
                        try:
                            # 确保帧大小准确
                            frame = audio_buffer[i:i + frame_size]
                            if len(frame) == frame_size:  # 确保完整帧
                                if not vad.is_speech(frame, vad_sample_rate):
                                    best_split = i
                                    break
                        except Exception as e:
                            logger.warning(f"VAD处理失败: {str(e)}")
                            continue

                    # 至少保留0.3秒音频继续处理
                    min_keep = int(0.3 * sample_rate * 2)
                    best_split = max(best_split, min_keep)

                    if best_split < len(audio_buffer):
                        segment = audio_buffer[:best_split]
                        remaining = audio_buffer[best_split:]
                        try:
                            audio = self._audio_processor.bytes_to_audio(segment, sample_rate)
                            audio = self._audio_processor.resample_audio(audio, sample_rate, TARGET_SAMPLE_RATE)
                            text = self._asr_pipeline(audio_in=audio)['text']

                            yield speech_pb2.SpeechResponse(
                                text=text,
                                is_final=False,
                                confidence=0.9
                            )

                            audio_buffer = remaining
                            last_recognition_time = current_time
                        except Exception as e:
                            logger.error(f"分段识别失败: {str(e)}")


    # ---------- 非流式识别 ----------
    def Recognize(self, request, context):
        try:
            if not request.audio_data:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("音频数据不能为空")
                return speech_pb2.SpeechResponse()

            sample_rate = request.sample_rate or TARGET_SAMPLE_RATE
            audio = self._audio_processor.bytes_to_audio(request.audio_data, sample_rate)
            if len(audio) == 0:
                return speech_pb2.SpeechResponse(text="", is_final=True)

            if sample_rate != TARGET_SAMPLE_RATE:
                audio = self._audio_processor.resample_audio(
                    audio, sample_rate, TARGET_SAMPLE_RATE
                )

            text = self._asr_pipeline(audio_in=audio)['text']
            return speech_pb2.SpeechResponse(
                text=text,
                is_final=True,
                confidence=0.95
            )
        except Exception as e:
            logger.error(f"识别错误: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"处理错误: {str(e)}")
            return speech_pb2.SpeechResponse()


# ---------- 启动 ----------
def serve(port=50051):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.max_send_message_length', 50 * 1024 * 1024)
        ]
    )
    speech_pb2_grpc.add_SpeechServicer_to_server(SpeechServicer(), server)
    server.add_insecure_port(f'[::]:{port}') # IP6
    server.add_insecure_port(f'0.0.0.0:{port}')  # IPv4
    server.start()
    logger.info(f"服务已启动，监听端口 {port}")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()