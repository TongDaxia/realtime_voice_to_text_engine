import argparse
import io
import logging
import time
import grpc
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import speech_pb2
import speech_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioStreamer:
    def __init__(self, audio_file, sample_rate=16000, chunk_duration=0.5):
        # 使用pydub加载多种音频格式
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        self.audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
        self.audio /= np.iinfo(np.int16).max  # 归一化
        self.sr = sample_rate
        self.chunk_size = int(self.sr * chunk_duration)
        self.position = 0
        self._iterator = self._generate_chunks()  # 创建生成器

    def _generate_chunks(self):
        """实际的生成器函数"""
        while self.position < len(self.audio):
            chunk = self.audio[self.position:self.position + self.chunk_size]
            self.position += self.chunk_size

            # 转换为WAV格式
            buf = io.BytesIO()
            sf.write(buf, chunk, self.sr, format='WAV', subtype='PCM_16')
            buf.seek(0)

            yield speech_pb2.SpeechRequest(
                audio_data=buf.getvalue(),
                sample_rate=self.sr,
                interim_results=(self.position < len(self.audio)))  # 非最后一块则为临时结果
            time.sleep(0.1)  # 模拟实时流

    def __iter__(self):
        """使类成为可迭代对象"""
        return self._iterator


def run_streaming_recognition(host, port, audio_file, language='zh-CN'):
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = speech_pb2_grpc.SpeechStub(channel)

    try:
        streamer = AudioStreamer(audio_file)
        responses = stub.StreamingRecognize(iter(streamer))  # 显式使用iter()

        for response in responses:
            if response.is_final:
                logger.info(f"最终结果: {response.text}")
                print(f"\033[92m最终结果: {response.text}\033[0m")  # 绿色显示
            else:
                logger.info(f"临时识别: {response.text}\r")
                print(f"临时识别: {response.text}", end='\r', flush=True)  # 行内刷新
    except grpc.RpcError as e:
        logger.error(f"RPC错误: {e.code().name}: {e.details()}")
    except Exception as e:
        logger.error(f"处理错误: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--language', default='zh-CN')
    args = parser.parse_args()

    run_streaming_recognition(args.host, args.port, args.audio, args.language)