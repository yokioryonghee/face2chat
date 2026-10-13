"""Offline rules by default; optional explicitly configured local Ollama."""
import ast
import json
import math
import operator
import os
import re
from urllib.request import Request,urlopen

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
def arithmetic(text):
    text=text.replace('더하기','+').replace('빼기','-').replace('곱하기','*').replace('나누기','/')
    match=re.fullmatch(r'\s*([0-9. +*/()\-]+)(?:는|은)?[?？]?\s*',text)
    if not match or len(match[1])>80: return None
    def value(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float): return node.value
        if isinstance(node,ast.BinOp) and type(node.op) in OPS: return OPS[type(node.op)](value(node.left),value(node.right))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub): return -value(node.operand)
        raise ValueError('Unsupported expression')
    try:
        answer=value(ast.parse(match[1],mode='eval').body)
        return f'{answer:g}' if math.isfinite(answer) else None
    except (SyntaxError,ValueError,ZeroDivisionError,OverflowError): return None

class ChatbotEngine:
    def __init__(self,model=None,transport=None):
        self.model=model or os.environ.get('FACE2CHAT_OLLAMA_MODEL')
        self.transport=transport or urlopen
    def generate_response(self,text,emotion='unknown',history=None):
        text=(text or '').strip()
        if not text: return '메시지를 입력하거나 녹음해 주세요.'
        if self.model:
            messages=[{'role':'system','content':'한국어로 간결하게 답하세요. 표정 분류는 불확실하며 감정이나 심리 상태를 단정하지 마세요.'}]
            messages.extend((history or [])[-12:])
            messages.append({'role':'user','content':text})
            request=Request('http://127.0.0.1:11434/api/chat',data=json.dumps({'model':self.model,'messages':messages,'stream':False}).encode(),headers={'Content-Type':'application/json'},method='POST')
            try:
                with self.transport(request,timeout=60) as response: result=json.load(response)
                answer=result['message']['content'].strip()
                if not answer: raise ValueError('Empty model response')
                return answer
            except Exception as exc:
                raise RuntimeError('Local Ollama request failed. Start Ollama and prepare the configured model, or use offline mode.') from exc
        answer=arithmetic(text)
        if answer is not None: return f'계산 결과는 {answer}입니다.'
        if any(word in text.lower() for word in ['안녕','hello','hi ']): return '안녕하세요. 무엇을 도와드릴까요?'
        if any(word in text for word in ['고마워','감사']): return '도움이 됐다니 다행이에요.'
        return '현재는 오프라인 규칙 응답 모드입니다. 간단한 계산과 인사를 처리할 수 있습니다. 자유로운 대화는 로컬 Ollama 모델을 설정해 주세요.'
    def respond(self,text,emotion=None): return self.generate_response(text,emotion or 'unknown')
