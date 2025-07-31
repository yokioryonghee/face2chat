# app.py
# ✅ 절대 경로 import 방식
import sys
import os

# 'modules' 폴더 자체를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

import gradio as gr
print(f"DEBUG: Gradio version in use: {gr.__version__}") # ⭐️ 이 줄 추가 ⭐️

from modules.pipeline import Face2ChatPipeline
from modules.emotion_detector import EmotionDetector
from modules.speech_to_text import SpeechToText
from modules.chatbot_engine import ChatbotEngine
from modules.text_to_speech import TextToSpeech
from modules.vision_analyzer import VisionAnalyzer # ⭐️ VisionAnalyzer 임포트 ⭐️

import numpy as np # numpy 임포트
import soundfile as sf # soundfile 임포트 (오디오 저장용)
import tempfile        # 임시 파일 생성용
import shutil          # 임시 파일 디렉토리 관리용 (현재는 직접 사용 안 함)


# 파이프라인 초기화
detector = EmotionDetector()
stt = SpeechToText()
bot = ChatbotEngine()
tts = TextToSpeech()
vision_analyzer = VisionAnalyzer() # ⭐️ VisionAnalyzer 인스턴스 생성 ⭐️
pipeline = Face2ChatPipeline(detector, stt, bot, tts, vision_analyzer) # ⭐️ pipeline에 전달 ⭐️

# Gradio에서 호출할 함수
def run_pipeline(image, audio):
    # Gradio가 제공하는 임시 파일 경로를 사용하여 STT 수행
    # audio는 (sample_rate, numpy_array) 튜플 형태 또는 파일 경로일 수 있음
    # 현재 speech_to_text.py는 파일 경로를 기대하므로, 튜플이라면 임시 파일로 저장
    audio_input_path = None

    if isinstance(audio, tuple): # audio가 (sample_rate, numpy_array) 튜플로 들어올 경우
        sr, audio_array = audio
        if audio_array is not None and audio_array.size > 0:
            try:
                # 임시 파일 생성 시 tempfile.mkdtemp() 사용 (gr.make_temp_dir() 경고 해결)
                # Gradio가 임시 디렉토리를 자동으로 관리하므로, 별도의 디렉토리를 만들 필요 없음
                # Gradio 4.0+ 버전에서는 gr.make_temp_dir()이 없어졌으므로 tempfile을 직접 사용.
                # 하지만 Gradio 내부적으로 임시파일을 생성하고 경로를 넘겨줄 가능성이 높으므로,
                # 여기서는 받은 경로를 그대로 사용하는 것을 우선시하고,
                # 튜플 형태의 오디오 입력만 파일로 변환합니다.
                
                # Gradio가 넘겨주는 오디오 튜플을 WAV 파일로 저장
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                    temp_audio_file = fp.name
                    sf.write(temp_audio_file, audio_array, sr)
                audio_input_path = temp_audio_file
                print(f"🎶 Gradio 튜플 오디오를 임시 WAV 파일로 저장: {audio_input_path}")
            except Exception as e:
                print(f"❗ Gradio 오디오 튜플을 파일로 저장 실패: {e}")
                audio_input_path = None # 오류 시 음성 인식 건너뛰기
        else:
            print("❗ 오디오 입력 (튜플)이 비어있거나 유효하지 않습니다.")
            audio_input_path = None
    elif isinstance(audio, str) and os.path.exists(audio): # audio가 파일 경로로 들어올 경우
        audio_input_path = audio
        print(f"🎶 Gradio 파일 경로 오디오 입력: {audio_input_path}")
    else:
        print("❗ 오디오 입력이 유효하지 않습니다.")
        audio_input_path = None

    emotion, text, response, audio_out_tuple = pipeline.run(image, audio_input_path)

    print("🚨 result from pipeline.run():", (emotion, text, response, "audio_out_tuple_exists")) # print audio_out as string to avoid large console output
    print("🚨 types:", [type(x) for x in (emotion, text, response, audio_out_tuple)])

    # pipeline.run에서 반환된 audio_out_tuple이 (np.ndarray, sample_rate) 형식인지 확인
    if not (isinstance(audio_out_tuple, tuple) and len(audio_out_tuple) == 2 and
            isinstance(audio_out_tuple[0], np.ndarray) and isinstance(audio_out_tuple[1], (int, float))):
        print("❗ pipeline.run에서 반환된 audio_out 형식이 잘못되었습니다. 무음 오디오로 대체합니다.")
        sample_rate = 44100
        silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
        audio_out_tuple = (silence, sample_rate) # (numpy_array, sample_rate) 형식으로 튜플 반환

    # Gradio에 반환하기 위해 (numpy_array, sample_rate) 튜플을 임시 파일로 저장
    final_audio_output_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            final_audio_output_path = fp.name
            sf.write(final_audio_output_path, audio_out_tuple[0], int(audio_out_tuple[1]))
        print(f"🎶 응답 오디오 임시 파일 저장: {final_audio_output_path}")
    except Exception as e:
        print(f"❗ 응답 오디오 파일 저장 중 오류 발생: {e}. 무음 오디오로 대체합니다.")
        sample_rate = 44100
        silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                final_audio_output_path = fp.name
                sf.write(final_audio_output_path, silence, sample_rate)
            print(f"🎶 오류 대체용 무음 오디오 임시 파일 저장: {final_audio_output_path}")
        except Exception as e_fallback:
            print(f"❗ 대체 오디오 파일 저장마저 실패: {e_fallback}")
            # 최후의 수단: 빈 오디오 파일 경로 반환 (UI에서 오디오 재생 안됨)
            final_audio_output_path = None
    
    # 임시 파일로 저장한 경우, 해당 파일 경로를 Gradio에 반환.
    # Gradio는 이 경로를 사용하여 웹 UI에서 오디오를 재생합니다.
    # Gradio 4.0+ 버전은 `type="filepath"`일 경우 파일을 자동으로 관리하므로,
    # 명시적인 `os.remove(temp_dir)`는 필요하지 않습니다.
    return emotion, text, response, final_audio_output_path


# 인터페이스 정의
interface = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Image(type="numpy", label="얼굴 이미지 (웹캠 입력)", streaming=True), # ⭐️ type을 "numpy"로 변경 ⭐️
        gr.Audio(type="numpy", label="음성 입력", streaming=True) # ⭐️ type을 "numpy"로 변경 ⭐️
    ],
    outputs=[
        gr.Textbox(label="감정"),
        gr.Textbox(label="음성 인식 결과"),
        gr.Textbox(label="챗봇 응답"),
        gr.Audio(label="응답 음성", type="filepath", autoplay=True) # ⭐️ TTS 출력 type을 "filepath"로 변경 ⭐️
    ],
    live=True, # ⭐️ live=True 추가 ⭐️
    allow_flagging="never", # ⭐️ 불필요한 플래그 방지 ⭐️
    title="Face2Chat: 감정 인식 음성 챗봇",
    description="웹캠과 마이크를 사용하여 감정을 인식하고 대화하는 챗봇입니다."
)

if __name__ == "__main__":
    # 모델 다운로드 확인 및 안내
    # Vosk 모델 경로 확인
    vosk_model_path = "models/vosk-model-small-en-us-0.15" # 또는 'models/vosk-model-ko-0.22'
    if not os.path.exists(vosk_model_path):
        print(f"\n[경고] Vosk 모델 '{vosk_model_path}'을(를) 찾을 수 없습니다.")
        print("Vosk 모델을 다운로드하여 'models' 폴더에 압축 해제해야 합니다.")
        print("예시: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
        print("또는 한국어 모델: https://alphacephei.com/vosk/models/vosk-model-ko-0.22.zip")
        print("다운로드 후 압축을 풀고, 압축 해제된 폴더 이름을 위 model_path와 동일하게 맞춰주세요.\n")
        # sys.exit(1) # 모델이 없으면 종료하도록 설정할 수 있습니다.

    # YOLOv8 모델 다운로드 확인 (vision_analyzer에서 처리되지만, 여기서도 안내 가능)
    # VisionAnalyzer 클래스 내에서 모델 다운로드를 처리하므로 여기서는 생략합니다.

    # Gradio 앱 실행
    interface.launch()