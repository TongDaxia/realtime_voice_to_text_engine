import argparse
import io
import logging
import time
import grpc
import librosa
from typing import Tuple
import numpy as np
import soundfile as sf
import speech_pb2
import speech_pb2_grpc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_audio_file(audio_file: str, sample_rate: int) -> Tuple[np.ndarray, int]:
    """更健壮的音频加载方法"""
    try:
        # 优先使用soundfile加载
        audio, sr = sf.read(audio_file)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return audio, sample_rate
    except Exception as sf_error:
        logger.warning(f"SoundFile加载失败，尝试Librosa: {str(sf_error)}")
        try:
            audio, sr = librosa.load(audio_file, sr=sample_rate)
            return audio, sr
        except Exception as librosa_error:
            logger.error(f"音频加载失败: {str(librosa_error)}")
            raise RuntimeError(f"无法加载音频文件: {audio_file}")


def streaming_recognize_file(stub, audio_file, language, sample_rate):
    """改进的流式识别方法"""
    try:
        # 加载音频文件
        audio, sr = load_audio_file(audio_file, sample_rate)
        logger.info(f"成功加载音频，时长: {len(audio) / sr:.2f}s, 采样率: {sr}Hz")

        def request_generator():
            chunk_size = sr * 1  # 1秒的音频数据
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]

                # 转换为WAV格式bytes
                buf = io.BytesIO()
                sf.write(buf, chunk, sr, format='WAV', subtype='PCM_16')

                yield speech_pb2.SpeechRequest(
                    audio_data=buf.getvalue(),
                    language=language,
                    sample_rate=sr,
                    interim_results=True
                )
                time.sleep(0.1)  # 模拟实时流

        # 发送请求并处理响应
        responses = stub.StreamingRecognize(request_generator())
        for i, response in enumerate(responses, 1):
            prefix = "临时结果" if not response.is_final else "最终结果"
            logger.info(f"{prefix} [{i}]: {response.text}")
            print(f"{prefix}: {response.text}")  # 控制台输出

    except grpc.RpcError as e:
        logger.error(f"gRPC通信错误: {e.code().name}: {e.details()}")
    except Exception as e:
        logger.error(f"处理错误: {str(e)}", exc_info=True)


def run(host, port, audio_file, streaming, language, sample_rate):
    """运行客户端"""
    try:
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = speech_pb2_grpc.SpeechStub(channel)

        if streaming:
            streaming_recognize_file(stub, audio_file, language, sample_rate)
        else:
            # 单次识别实现...
            pass

    except Exception as e:
        logger.error(f"客户端运行失败: {str(e)}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='语音识别客户端')
    parser.add_argument('--host', default='localhost', help='服务端地址')
    parser.add_argument('--port', type=int, default=50051, help='服务端端口')
    parser.add_argument('--audio', required=True, help='音频文件路径')
    parser.add_argument('--streaming', action='store_true', help='使用流式识别')
    parser.add_argument('--language', default='zh-CN', help='语言代码')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='音频采样率(Hz)')

    args = parser.parse_args()
    run(args.host, args.port, args.audio, args.streaming,
        args.language, args.sample_rate)