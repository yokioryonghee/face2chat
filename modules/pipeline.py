# modules/pipeline.py
import numpy as np # np 임포트 추가 (STT 길이 체크에 필요)

# ⭐️ 추가: VisionAnalyzer 및 다른 모듈들을 명시적으로 임포트 (타입 힌트 및 린터 경고 해결용) ⭐️
from .emotion_detector import EmotionDetector
from .speech_to_text import SpeechToText
from .chatbot_engine import ChatbotEngine
from .text_to_speech import TextToSpeech
from .vision_analyzer import VisionAnalyzer # ⭐️ VisionAnalyzer 명시적 임포트 ⭐️


class Face2ChatPipeline:
    # ⭐️ 수정: __init__ 메서드에 타입 힌트 추가 및 vision_analyzer 인자 포함 ⭐️
    def __init__(self, detector: EmotionDetector, stt: SpeechToText, bot: ChatbotEngine, tts: TextToSpeech, vision_analyzer: VisionAnalyzer):
        self.detector = detector
        self.stt = stt
        self.bot = bot
        self.tts = tts
        self.vision_analyzer = vision_analyzer # ⭐️ 초기화 ⭐️

    def run(self, image, audio):
        # 1. 감정 인식
        # image는 numpy 배열 (Gradio Image type="numpy"로 설정했으므로)
        emotion = self.detector.detect(image)
        
        # 2. 주변 상황 분석 (새로운 기능)
        scene_info = self.vision_analyzer.analyze_scene(image) # ⭐️ 추가 ⭐️
        print(f"[파이프라인] 주변 상황 분석 결과: {scene_info}")

        # 3. 음성 인식
        # audio는 STT 모듈이 기대하는 파일 경로 (app.py에서 처리됨)
        text = self.stt.transcribe(audio)

        # STT 결과 예외 처리 로직
        min_text_length = 3
        if not text or len(text.strip()) < min_text_length:
            print(f"[파이프라인] STT 결과가 비어있거나 너무 짧습니다: '{text}'. 기본 응답으로 대체합니다.")
            
            # STT 실패 시 챗봇에게 보낼 텍스트에 주변 상황 정보를 포함 (선택적)
            if scene_info and "주변에서 다음을 감지했습니다" in scene_info:
                # scene_info에서 객체 목록만 추출하여 사용자에게 더 직접적인 피드백 제공
                scene_objects = scene_info.split("주변에서 다음을 감지했습니다:")[1].strip().replace('.', '')
                response_text = f"잘 이해하지 못했어요. 혹시 주변의 {scene_objects}과(와) 관련된 질문인가요?"
            else:
                response_text = "잘 이해하지 못했어요. 다시 말씀해 주시겠어요?"

            audio_out = self.tts.synthesize(response_text)
            return emotion, text, response_text, audio_out

        # 4. 챗봇 응답 생성
        # 챗봇에 감정 정보와 함께 주변 상황 정보를 전달하여 응답을 생성하도록 프롬프트를 구성
        # 챗봇에 전달할 최종 텍스트를 구성 (기존 사용자 발화 + 주변 상황 정보)
        # chatbot_engine의 generate_response가 text와 emotion만 받으므로,
        # 여기서는 scene_info를 text에 통합하여 전달.
        full_text_for_chatbot = f"{scene_info}. 사용자가 말했어요: '{text}'" if scene_info else text
        
        # ⭐️ chatbot_engine.py의 메서드 이름을 'generate_response'로 수정 ⭐️
        response = self.bot.generate_response(full_text_for_chatbot, emotion) 
        
        # 5. 텍스트를 음성으로 변환
        audio_out = self.tts.synthesize(response)
        
        return emotion, text, response, audio_out