import torch
import torch.nn as nn
import os
from config import MODEL_WEIGHTS_PATH, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS

class SleepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SleepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return self.sigmoid(out)

class SleepDetectorModel:
    def __init__(self):
        # 👇 가져온 공통 변수를 사용합니다.
        self.model = SleepLSTM(
            input_size=INPUT_SIZE, 
            hidden_size=HIDDEN_SIZE, 
            num_layers=NUM_LAYERS
        )
        self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
        self.model.eval()
        # 가중치 파일이 존재할 경우에만 로드 (초기 테스트 에러 방지)
        if os.path.exists(MODEL_WEIGHTS_PATH):
            self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
            print(f"[AI] Model loaded successfully from {MODEL_WEIGHTS_PATH}")
        else:
            print(f"[AI] Warning: Model weights not found at {MODEL_WEIGHTS_PATH}. Using untrained model.")
            
        self.model.eval()

    def predict(self, input_data):
        import torch # 상단에 없다면 안전하게 함수 안에서라도 호출

    # 1. 데이터 길이가 충분한지 다시 한번 확인 (예: 30프레임)
        if len(input_data) < 30:
            return False

        # 2. 들어온 데이터를 [EAR, MAR, Pitch] 형태의 리스트로 정제
        processed_data = []
        for d in input_data:
            # 딕셔너리 형태일 경우 값만 추출
            if isinstance(d, dict):
                processed_data.append([d.get('ear', 0), d.get('mar', 0), d.get('pitch', 0)])
            else:
                processed_data.append(d)

        # 3. 누락되었던 input_tensor 명시적으로 생성 (배치 차원 1 추가)
        input_tensor = torch.tensor(processed_data, dtype=torch.float32).unsqueeze(0)

        # 4. 모델 예측 진행
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = output.item()
            
            print(f"👉 현재 수면 확률: {probability:.4f}") 
            
        return probability > 0.5