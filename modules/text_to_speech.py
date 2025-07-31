from gtts import gTTS
import numpy as np
import soundfile as sf
import tempfile
import os

class TextToSpeech:
    def synthesize(self, text): # speak -> synthesize로 변경
        """
        텍스트를 음성으로 변환하여 Gradio가 요구하는 형식 (numpy.ndarray, sample_rate) 으로 반환합니다.
        gTTS는 MP3를 생성하므로 임시 파일을 사용하여 변환 후 numpy 배열로 읽어옵니다.
        """
        if not text:
            sample_rate = 44100
            duration_sec = 0.5
            silence = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
            print("No text for TTS. Returning silence.")
            return silence, sample_rate

        temp_audio_file = None

        try:
            # 언어 설정: 'KO' -> 'ko'로 수정
            tts = gTTS(text=text, lang='ko') # ⭐️ 'KO' -> 'ko'로 수정 ⭐️

            # 임시 파일에 저장
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp_mp3:
                tts.save(fp_mp3.name)
                temp_audio_file = fp_mp3.name

            # 사운드 파일 로딩 (MP3를 numpy 배열로)
            audio_data, sample_rate = sf.read(temp_audio_file)
            audio_data = audio_data.astype(np.float32) # Gradio는 float32를 선호

            # SF.read는 스테레오 오디오의 경우 (N, 2) 형태를 반환할 수 있으므로, 모노로 변환
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            print(f"Synthesized audio with sample rate: {sample_rate}, shape: {audio_data.shape}")
            return audio_data, sample_rate

        except Exception as e:
            print(f"Error in TextToSpeech synthesis: {e}")
            sample_rate = 44100
            duration_sec = 1.0
            silence = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
            print("TTS failed. Returning silence.")
            return silence, sample_rate

        finally:
            if temp_audio_file and os.path.exists(temp_audio_file):
                os.remove(temp_audio_file) # 임시 MP3 파일 정리