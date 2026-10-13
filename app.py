"""UI construction does not initialize/download optional ML models."""
import os
from pathlib import Path
from modules.chatbot_engine import ChatbotEngine

ROOT=Path(__file__).resolve().parent

def respond(text,image=None,audio=None,use_vision=False,use_speech=False,use_tts=False,history=None):
    emotion,scene='unknown',''
    if use_speech:
        if not audio: raise ValueError('Record or upload audio before enabling speech recognition')
        from modules.speech_to_text import SpeechToText
        model_path=os.environ.get('FACE2CHAT_VOSK_MODEL',str(ROOT/'models/vosk-model-small-en-us-0.15'))
        text=SpeechToText(model_path).transcribe(audio)
    if use_vision:
        from modules.emotion_detector import EmotionDetector
        from modules.vision_analyzer import VisionAnalyzer
        emotion=EmotionDetector().detect(image)
        scene=VisionAnalyzer().analyze_scene(image)
    bot=ChatbotEngine()
    # Keep the user's utterance intact in offline mode; scene is shown separately.
    prompt=f'{text}\n사진의 객체 탐지 결과(불확실): {scene}' if bot.model and scene else text
    answer=bot.generate_response(prompt,emotion,history)
    output=None
    if use_tts:
        from modules.text_to_speech import TextToSpeech
        output=TextToSpeech().synthesize(answer)
    updated=list(history or [])+[{'role':'user','content':text or ''},{'role':'assistant','content':answer}]
    return emotion,scene,answer,output,updated[-12:]

def build_app():
    import gradio as gr
    def handle(*args):
        try: return respond(*args)
        except Exception as exc: raise gr.Error(str(exc)) from exc
    with gr.Blocks() as demo:
        gr.Markdown('# Face2Chat\n텍스트 모드는 모델 다운로드 없이 실행됩니다. 영상·음성 기능은 아래에서 직접 켜세요.')
        history=gr.State([])
        text=gr.Textbox(label='메시지')
        image=gr.Image(type='numpy',label='사진 (선택)')
        audio=gr.Audio(type='filepath',label='음성 (선택)')
        vision=gr.Checkbox(label='표정·객체 분석 사용',value=False)
        speech=gr.Checkbox(label='Vosk 음성 인식 사용',value=False)
        tts=gr.Checkbox(label='gTTS 응답 음성 사용 (인터넷 필요)',value=False)
        emotion=gr.Textbox(label='표정 분류 (실제 감정의 판정이 아님)')
        scene=gr.Textbox(label='객체 탐지')
        answer=gr.Textbox(label='응답')
        output=gr.Audio(label='응답 음성')
        gr.Button('보내기').click(handle,[text,image,audio,vision,speech,tts,history],[emotion,scene,answer,output,history])
        gr.Button('대화 초기화').click(lambda:[],outputs=history)
    return demo

if __name__=='__main__': build_app().launch()
