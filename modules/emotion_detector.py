# face2chat - 감정 인식 음성 챗봇
# 프로젝트 파이프라인 구조 설계

# 프로젝트 디렉토리 구조
# face2chat/
# └── app.py                # Gradio 인터페이스 진입점
#     modules/
#     ├── emotion_detector.py  # 얼굴 감정 인식 모듈
#     ├── speech_to_text.py     # 음성 인식 모듈
#     ├── chatbot_engine.py    # GPT 응답 생성 모듈
#     ├── text_to_speech.py     # TTS 음성 생성 모듈
#     └── pipeline.py          # 전체 플로우 제어 모듈

# 클래스 정의 개인 프로젝트 바이너 구조

from deepface import DeepFace
import cv2
import numpy as np


class EmotionDetector:
    def __init__(self):
        print("(감정 인식기) 초기화 완료")

    def detect(self, image):
        # image가 None (웹캠이 비활성화되었거나 초기 입력이 없는 경우)
        if image is None:
            print("[감정 인식기] 이미지 입력이 없습니다.")
            return "알 수 없음"
            
        if isinstance(image, str): # Gradio Image(type="filepath")
            img = cv2.imread(image)
        elif isinstance(image, np.ndarray): # Gradio Image(type="numpy")
            img = image
        else:
            print("[오류] 이미지 입력 형식이 잘못되었습니다.")
            return "알 수 없음"

        # 이미지가 로드되지 않았거나 비어 있는 경우
        if img is None or img.size == 0:
            print("[감정 인식기] 처리할 이미지가 비어 있습니다.")
            return "알 수 없음"

        try:
            # enforce_detection=False: 얼굴을 찾지 못해도 오류를 발생시키지 않음
            # 대신 빈 리스트를 반환할 수 있음
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True) # silent=True 추가로 콘솔 출력 줄임
            
            if result and len(result) > 0:
                emotion = result[0]['dominant_emotion']
                print(f"(감정 인식기) 감정 분석 결과: {emotion}")
                return emotion
            else:
                print("(감정 인식기) 얼굴 감지 실패 또는 감정 분석 결과 없음.")
                return "알 수 없음"
        except Exception as e:
            print(f"(감정 인식 오류) {e}")
            return "감정 인식 실패"