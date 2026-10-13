# modules/speech_to_text.py

import wave
import json
import os

import soundfile as sf
import numpy as np
import tempfile
from resampy import resample # resampy 임포트

class SpeechToText:
    def __init__(self, model_path="models/vosk-model-small-en-us-0.15"):
        print("[Vosk STT] 초기화 중...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk 모델을 찾을 수 없습니다: {model_path}")
        from vosk import Model
        self.model = Model(model_path)
        print("[Vosk STT] 초기화 완료.")

    def transcribe(self, audio_path):
        print("[STT 디버그] 오디오 경로:", audio_path)
        if not audio_path or not os.path.exists(audio_path): # audio_path가 None이거나 존재하지 않는 경우 처리
            print("[STT 오류] 오디오 파일 경로가 유효하지 않습니다.")
            return "" # 빈 문자열 반환

        from vosk import KaldiRecognizer
        temp_wav_file = None
        wf = None # wave 파일 객체 초기화
        try:
            # soundfile을 사용하여 오디오 파일 읽기
            # dtype='float32'로 지정하여 데이터를 float32로 읽어옵니다.
            audio_data, samplerate = sf.read(audio_path, dtype='float32')

            # Vosk는 16kHz, 1채널, 16비트 PCM 형식을 선호합니다.

            # 모노 채널로 변환 (스테레오인 경우)
            if audio_data.ndim > 1:
                print("[STT 경고] 스테레오 오디오: 모노로 변환")
                audio_data = audio_data.mean(axis=1) # 평균으로 모노 변환

            # 샘플 레이트 변환 (필요시)
            if samplerate != 16000:
                print(f"[STT 경고] 샘플 레이트 불일치: {samplerate} -> 16000으로 변환 시도")
                # resampy.resample 사용
                audio_data = resample(audio_data, samplerate, 16000)
                samplerate = 16000
                print(f"[STT 경고] 리샘플링 완료. 새 샘플 레이트: {samplerate}")


            # 16비트 정수형으로 변환
            # float32 (-1.0 to 1.0) -> int16 (-32768 to 32767) 스케일링
            # audio_data는 이미 float32이므로 단순히 스케일링 후 astype(np.int16)
            audio_data_int16 = (np.clip(audio_data,-1,1) * 32767).astype(np.int16)

            import io
            buffer=io.BytesIO()
            sf.write(buffer,audio_data_int16,samplerate,format='WAV',subtype='PCM_16')
            buffer.seek(0)
            wf=wave.open(buffer,'rb')
        except Exception as exc:
            raise RuntimeError('Audio conversion failed') from exc

        # Vosk 모델이 요구하는 최종 WAV 형식 확인 (선택 사항, 이미 변환했으므로 통과되어야 함)
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print(f"[STT 오류] 최종 wav 형식 문제: 채널={wf.getnchannels()}, 비트={wf.getsampwidth()*8}, 압축={wf.getcomptype()}")
            # wf 객체가 유효하면 닫아줍니다.
            if wf:
                wf.close()
            return "" # 빈 문자열 반환

        rec = KaldiRecognizer(self.model, wf.getframerate())
        result = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                partial = json.loads(rec.Result())
                result += partial.get("text", "") + " "
        partial = json.loads(rec.FinalResult())
        result += partial.get("text", "")

        wf.close() # wave 파일 닫기
        print(f"[STT 결과] {result.strip()}")
        return result.strip()
