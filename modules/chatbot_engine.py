# modules/chatbot_engine.py
class ChatbotEngine:
    def __init__(self):
        # 향후 GPT API 키나 세션 초기화가 필요한 경우 여기에 작성
        pass

    def generate_response(self, text: str, emotion: str) -> str:
        # 빈 입력 처리
        if not text.strip():
            return "음성을 잘 못 들었어요. 다시 말씀해 주세요."

        # ⭐️ pipeline.py에서 text는 이미 scene_info와 사용자의 발화가 합쳐진 상태일 수 있음 ⭐️
        # 예: "주변에서 다음을 감지했습니다: person, laptop. 사용자가 말했어요: '안녕하세요?'"
        # 이 합쳐진 텍스트를 파싱하여 활용하거나, LLM에 직접 전달하여 문맥을 이해하게 합니다.

        response_prefix = {
            "happy": "기분이 좋아 보여요! ",
            "sad": "기운 내세요. ",
            "angry": "마음을 진정시켜볼까요? ",
            "surprise": "놀라셨군요! ",
            "fear": "걱정하지 마세요. ",
            "disgust": "불쾌하셨군요... ",
            "neutral": "",
            "unknown": ""
        }

        prefix = response_prefix.get(emotion, "")
        
        # ⭐️ LLM (GPT) 연동 코드 예시 (실제 API 호출 코드로 교체 필요) ⭐️
        # 현재는 단순한 문자열 조합이지만, 여기에 OpenAI, Google Gemini 등의 API 호출 로직이 들어갑니다.
        # 이 때, 감정 정보와 주변 상황 정보(text에 포함됨)를 프롬프트에 잘 넣어주면 더 똑똑한 챗봇이 됩니다.
        
        # 예시: OpenAI API를 사용하는 경우 (주석 처리됨)
        # from openai import OpenAI
        # client = OpenAI(api_key="YOUR_OPENAI_API_KEY") # 실제 API 키로 교체

        # try:
        #     system_prompt = "당신은 사용자의 감정과 주변 상황을 고려하여 친절하고 공감적인 대화를 나누는 챗봇입니다. 짧고 간결하게 대화하세요."
        #     # text에 scene_info와 사용자 발화가 합쳐져 있으므로 이를 분리하여 프롬프트에 활용
        #     if "주변에서 다음을 감지했습니다:" in text:
        #         scene_info_for_llm = text.split("주변에서 다음을 감지했습니다:")[1].split('.')[0].strip()
        #         user_text_for_llm = text.split('.', 1)[1].replace("사용자가 말했어요:", "").strip() if len(text.split('.', 1)) > 1 else ""
        #     else:
        #         scene_info_for_llm = "알 수 없음"
        #         user_text_for_llm = text
            
        #     user_message = f"사용자의 감정: {emotion}. 주변 상황: {scene_info_for_llm}. 사용자가 말한 내용: {user_text_for_llm}."
        #     messages = [
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_message}
        #     ]
        #     completion = client.chat.completions.create(
        #         model="gpt-3.5-turbo", # 또는 다른 모델
        #         messages=messages
        #     )
        #     llm_response = completion.choices[0].message.content
        #     response = f"{prefix}{llm_response}"
        # except Exception as e:
        #     print(f"(챗봇 엔진 오류) LLM 응답 생성 실패: {e}")
        #     response = f"{prefix}당신은 이렇게 말했어요: '{text}'" # LLM 실패 시 대체 응답
        
        # 현재는 LLM 연동 없이 단순 문자열 조합
        response = f"{prefix}당신은 이렇게 말했어요: '{text}'"

        return response
    
    # 이 respond 메서드는 pipeline.py에서 generate_response로 호출되므로
    # 사용되지 않거나 다른 목적으로 사용될 수 있습니다.
    # 현재 구조에서는 generate_response가 주된 응답 생성 로직입니다.
    def respond(self, text, emotion=None):
        if emotion:
            print(f"(챗봇) 감정 고려하여 응답 생성 중... 감정: {emotion}")
            return f"({emotion} 상태에서) 당신은 이렇게 말했어요: '{text}'"
        else:
         return f"당신은 이렇게 말했어요: '{text}'"