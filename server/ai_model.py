import torch
import torch.nn as nn
import os
from config import (
    WINDOW_SIZE, MODEL_WEIGHTS_PATH, 
    MODEL_INPUT_SIZE, MODEL_HIDDEN_SIZE, MODEL_NUM_LAYERS
)

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
        self.seq_length = WINDOW_SIZE
        self.model = SleepLSTM(
            input_size=MODEL_INPUT_SIZE, 
            hidden_size=MODEL_HIDDEN_SIZE, 
            num_layers=MODEL_NUM_LAYERS
        )
        
        # 가중치 파일이 존재할 경우에만 로드 (초기 테스트 에러 방지)
        if os.path.exists(MODEL_WEIGHTS_PATH):
            self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
            print(f"[AI] Model loaded successfully from {MODEL_WEIGHTS_PATH}")
        else:
            print(f"[AI] Warning: Model weights not found at {MODEL_WEIGHTS_PATH}. Using untrained model.")
            
        self.model.eval()

    def predict(self, timeseries_buffer):
        if len(timeseries_buffer) < self.seq_length:
            return False
            
        features = []
        for frame in timeseries_buffer[-self.seq_length:]:
            if not frame.get('face_detected', True):
                return True 
            features.append([frame.get('ear', 0), frame.get('mar', 0), frame.get('pitch', 0)])
            
        input_tensor = torch.tensor([features], dtype=torch.float32)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = output.item()
            
        return probability > 0.5