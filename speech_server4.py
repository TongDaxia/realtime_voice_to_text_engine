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
PUNC_MODEL = 'damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
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

    def _get_punctuation_model(self):
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
        if not text.strip():
            return text
        try:
            punc_model = self._get_punctuation_model()
            result = punc_model(text)
            return result[0]['text'] if isinstance(result, list) else text + "。"
        except Exception as e:
            logger.warning(f"标点预测失败: {str(e)}")
            return text + "。"

    # ---------- 流式识别 ----------
    def StreamingRecognize(self, request_iterator, context):
        audio_buffer = bytearray()
        sample_rate = None

        for request in request_iterator:
            if sample_rate is None:
                sample_rate = request.sample_rate
                logger.info(f"开始流式识别，采样率: {sample_rate} Hz")

            # 空包：客户端表示结束
            if not request.audio_data:
                logger.info("收到空包，开始最终识别...")
                if len(audio_buffer) == 0:
                    yield speech_pb2.SpeechResponse(text="", is_final=True)
                    break

                try:
                    audio = self._audio_processor.bytes_to_audio(
                        audio_buffer, sample_rate
                    )
                    audio = self._audio_processor.resample_audio(
                        audio, sample_rate, TARGET_SAMPLE_RATE
                    )
                    text = self._asr_pipeline(audio_in=audio)['text']
                    yield speech_pb2.SpeechResponse(
                        text=self._add_punctuation(text),
                        is_final=True,
                        confidence=0.95
                    )
                except Exception as e:
                    logger.error(f"最终识别失败: {str(e)}")
                    yield speech_pb2.SpeechResponse(text="", is_final=True)
                finally:
                    audio_buffer.clear()
                break

            # 累积数据
            audio_buffer.extend(request.audio_data)

            # 中间结果（可选）
            if request.interim_results and len(audio_buffer) > 1024:
                try:
                    audio = self._audio_processor.bytes_to_audio(
                        audio_buffer, sample_rate
                    )
                    audio = self._audio_processor.resample_audio(
                        audio, sample_rate, TARGET_SAMPLE_RATE
                    )
                    text = self._asr_pipeline(audio_in=audio)['text']
                    yield speech_pb2.SpeechResponse(
                        text=text,
                        is_final=False,
                        confidence=0.9
                    )
                except Exception as e:
                    logger.warning(f"中间结果失败: {str(e)}")

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
                text=self._add_punctuation(text),
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