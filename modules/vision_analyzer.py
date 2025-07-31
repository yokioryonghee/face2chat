# modules/vision_analyzer.py
from ultralytics import YOLO
import cv2
import numpy as np
import os # 파일 경로 확인용

class VisionAnalyzer:
    def __init__(self, model_path="yolov8n.pt"):
        print(f"(시각 분석기) {model_path} 모델 초기화 중...")
        self.model = None
        try:
            # YOLOv8 모델 로드
            # 모델 파일이 없으면 자동으로 다운로드 시도 (인터넷 연결 필요)
            # models 디렉토리에 있는지 확인하고 없으면 다운로드.
            # ultralytics는 기본적으로 캐시 디렉토리에 다운로드합니다.
            self.model = YOLO(model_path)
            print("(시각 분석기) YOLO 모델 초기화 완료.")
        except Exception as e:
            self.model = None
            print(f"[오류] YOLO 모델 로드 실패: {e}")
            print("YOLO 모델을 수동으로 다운로드해야 할 수 있습니다.")
            print("다음 링크에서 'yolov8n.pt' 파일을 다운로드하여 'models' 폴더에 넣어주세요:")
            print("https://github.com/ultralytics/ultralytics/releases/download/v8.2.0/yolov8n.pt")


    def analyze_scene(self, image):
        if self.model is None:
            return "시각 분석기 모델이 로드되지 않았습니다."

        img_to_process = None
        if isinstance(image, str): # Gradio Image(type="filepath")
            if os.path.exists(image):
                img_to_process = cv2.imread(image)
            else:
                print(f"[시각 분석기] 이미지 파일이 존재하지 않습니다: {image}")
                return "이미지 분석 실패: 파일 없음"
        elif isinstance(image, np.ndarray): # Gradio Image(type="numpy")
            # Gradio에서 넘겨주는 이미지는 이미 numpy 배열이므로 직접 사용
            img_to_process = image
        elif image is None: # 웹캠 연결이 안 되어 이미지가 None으로 들어오는 경우
            print("[시각 분석기] 이미지 입력이 없습니다 (초기 로드 또는 웹캠 비활성화).")
            return "이미지 분석 실패: 입력 없음"
        else:
            print("[오류] 이미지 입력 형식이 잘못되었습니다.")
            return "이미지 분석 실패: 잘못된 입력 형식"

        # 이미지 로드 실패 또는 빈 이미지 처리
        if img_to_process is None or img_to_process.size == 0:
            print("[시각 분석기] 처리할 이미지가 비어 있습니다.")
            return "이미지 분석 실패: 빈 이미지"

        try:
            # YOLOv8 모델로 객체 감지
            # predict 함수는 리스트를 반환할 수 있으므로 첫 번째 결과를 가져옵니다.
            # conf=0.5는 신뢰도 임계값으로, 50% 미만 신뢰도 객체는 무시합니다.
            # verbose=False로 설정하여 predict 함수의 자세한 콘솔 출력을 줄일 수 있습니다.
            results = self.model.predict(img_to_process, conf=0.5, verbose=False)
            
            # 감지된 객체 정보 파싱
            detected_objects = []
            for r in results:
                # r.boxes는 감지된 바운딩 박스 객체를 포함합니다.
                # r.boxes.cls는 각 박스의 클래스 인덱스를 numpy 배열로 반환합니다.
                for c in r.boxes.cls: 
                    class_name = self.model.names[int(c)] # 클래스 인덱스를 이름으로 매핑
                    detected_objects.append(class_name)
            
            if detected_objects:
                # 감지된 객체들의 중복을 제거하고 쉼표로 연결
                unique_objects = list(set(detected_objects))
                scene_description = "주변에서 다음을 감지했습니다: " + ", ".join(unique_objects) + "."
                return scene_description
            else:
                return "주변에서 특별한 것을 감지하지 못했습니다."
        except Exception as e:
            print(f"[시각 분석기 오류] 객체 감지 실패: {e}")
            return "이미지 분석 실패: 처리 오류"