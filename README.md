# Face2Chat

텍스트 응답에 선택적으로 표정 분류, 객체 탐지, 음성 인식과 응답 음성을 연결하는 Gradio 앱입니다.

## 기본 실행

Python 3.11 이상을 사용합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Windows에서는 `.\.venv\Scripts\Activate.ps1`로 활성화합니다. 기본 텍스트 모드는 모델 다운로드 없이 실행되며 인사와 간단한 계산을 처리합니다. 일반 지식 질문을 답하는 LLM으로 설명하지 않습니다.

## 로컬 LLM 대화

Ollama를 별도로 설치하고 사용할 모델을 준비한 뒤 `FACE2CHAT_OLLAMA_MODEL`에 모델 이름을 지정합니다. 앱은 `127.0.0.1:11434/api/chat`으로 대화 요청을 보내고 사용자 세션의 최근 메시지를 전달합니다. 설정하지 않으면 네트워크 요청 없이 규칙 응답을 사용합니다.

```bash
FACE2CHAT_OLLAMA_MODEL=your-installed-model python app.py
```

API 형식: https://docs.ollama.com/api/chat

## 영상·음성 기능

```bash
python -m pip install -r requirements-multimodal.txt
```

앱에서 필요한 기능을 직접 켭니다. 해당 옵션을 선택하기 전에는 모델을 초기화하지 않습니다.

- 표정/객체 분석: DeepFace와 YOLO의 첫 모델 다운로드가 필요합니다. 사진은 RGB에서 해당 도구의 BGR 입력으로 변환합니다. 얼굴을 찾지 못하면 감정을 추정하지 않습니다. 표정 분류는 실제 감정이나 심리 상태의 판정이 아닙니다.
- 음성 인식: Vosk의 해당 언어 모델을 직접 내려받아 압축을 풀고 `FACE2CHAT_VOSK_MODEL`에 폴더 경로를 지정합니다. 기본 경로는 `models/vosk-model-small-en-us-0.15`이며 영어 모델입니다. 한국어는 한국어 Vosk 모델을 선택해야 합니다. 모델: https://alphacephei.com/vosk/models
- 응답 음성: gTTS는 인터넷에 텍스트를 전송해 음성을 생성합니다. 이 기능은 기본적으로 꺼져 있습니다.

입력 녹음/사진, 모델과 개발 환경은 Git에 추가하지 않습니다. 모델 누락이나 연결 실패는 UI 오류로 표시합니다.

## 테스트

```bash
python -m pip install pytest soundfile resampy gTTS opencv-python-headless
python -m pytest -q
```

기본 앱 생성, 오프라인 응답, mock 로컬 LLM 요청, Vosk 모델 누락과 mock 음성 변환을 검사합니다. 실제 모델의 인식 품질이나 자유 대화 품질을 입증하는 테스트는 아닙니다.
