import io
import json
from pathlib import Path
import numpy as np
import pytest
from modules.chatbot_engine import ChatbotEngine

def test_offline_mode_responds_without_echo_or_network(monkeypatch):
    monkeypatch.delenv('FACE2CHAT_OLLAMA_MODEL',raising=False)
    bot=ChatbotEngine(transport=lambda *a,**k:pytest.fail('unexpected network'))
    assert bot.generate_response('2 더하기 2는?')=='계산 결과는 4입니다.'
    assert '안녕하세요' in bot.generate_response('안녕하세요')
    assert '오프라인' in bot.generate_response('지구의 역사를 설명해줘')
    assert '__import__' not in bot.generate_response("__import__('os').system('echo bad')")

def test_local_llm_transport_and_history():
    calls=[]
    def transport(request,timeout):
        calls.append(request)
        return io.BytesIO(json.dumps({'message':{'content':'실제 모델 응답'}}).encode())
    bot=ChatbotEngine(model='test-only',transport=transport)
    assert bot.generate_response('안녕',history=[{'role':'user','content':'지난 질문'}])=='실제 모델 응답'
    body=json.loads(calls[0].data)
    assert body['stream'] is False and body['messages'][-2]['content']=='지난 질문'
    assert calls[0].full_url=='http://127.0.0.1:11434/api/chat'

def test_text_ui_does_not_require_models(monkeypatch):
    monkeypatch.delenv('FACE2CHAT_OLLAMA_MODEL',raising=False)
    from app import respond,build_app
    result=respond('2+2')
    assert result[2]=='계산 결과는 4입니다.' and result[3] is None
    assert len(result[4])==2
    app=build_app()
    assert app is not None

def test_missing_vosk_model_has_clear_error(tmp_path):
    from modules.speech_to_text import SpeechToText
    with pytest.raises(FileNotFoundError): SpeechToText(str(tmp_path/'absent'))

def test_audio_is_consumed_before_cleanup(tmp_path,monkeypatch):
    import sys,types,soundfile as sf
    from modules.speech_to_text import SpeechToText
    model=tmp_path/'model'; model.mkdir()
    wav=tmp_path/'speech.wav'; sf.write(wav,np.zeros(5000,dtype=np.float32),22050)
    class Recognizer:
        def __init__(self,model,rate): assert rate==16000; self.bytes=0
        def AcceptWaveform(self,data): self.bytes+=len(data); return False
        def FinalResult(self): assert self.bytes>0; return json.dumps({'text':'테스트'})
    monkeypatch.setitem(sys.modules,'vosk',types.SimpleNamespace(Model=lambda p:object(),KaldiRecognizer=Recognizer))
    assert SpeechToText(str(model)).transcribe(str(wav))=='테스트'
