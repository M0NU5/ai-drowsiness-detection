import numpy as np

class SleepDetectorModel:
    def __init__(self):
        # 실제 환경에서는 여기서 data/weights/lstm_model.pth 등을 로드합니다.
        self.model_loaded = True 

    def predict(self, timeseries_buffer):
        """
        timeseries_buffer: N프레임 분량의 데이터 (리스트 또는 numpy array)
        반환값: True(수면) / False(정상)
        """
        # TODO: 실제 LSTM 모델 추론 로직 삽입
        # 현재는 더미 로직: 마지막 프레임의 EAR이 0.2 이하가 연속되면 수면으로 간주
        if len(timeseries_buffer) == 0:
            return False
            
        latest_data = timeseries_buffer[-1]
        
        # 얼굴을 놓친 경우 이탈(수면)로 판단
        if not latest_data.get('face_detected', True):
            return True
            
        ear = latest_data.get('ear', 1.0)
        
        # 단순 임계값 테스트용 (실제로는 버퍼 전체를 모델에 입력)
        if ear < 0.2:
            return True
        return False