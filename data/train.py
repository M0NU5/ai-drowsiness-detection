import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# --- 1. 하이퍼파라미터 설정 ---
SEQ_LENGTH = 30     # 30프레임씩 묶어서 판단 (약 1~2초)
INPUT_SIZE = 3      # 입력 피처 수 (EAR, MAR, Pitch)
HIDDEN_SIZE = 64    # LSTM 은닉층 크기
NUM_LAYERS = 2      # LSTM 층 수
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# --- 2. 모델 정의 (LSTM) ---
class SleepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SleepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # 0(정상) ~ 1(수면) 사이의 확률 출력
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        # 30개 프레임 중 가장 마지막 시점의 결과만 뽑아서 판단
        out = self.fc(out[:, -1, :]) 
        return self.sigmoid(out)

# --- 3. 데이터 준비 함수 (Sliding Window) ---
def create_sequences(data, labels, seq_length):
    xs, ys = [], []
    # 데이터 길이가 30프레임도 안 되면 스킵
    if len(data) <= seq_length:
        return np.array(xs), np.array(ys)
        
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = labels[i + seq_length] # 30프레임 묶음의 마지막 순간 라벨을 정답으로 씀
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model():
    # 폴더 내의 모든 CSV 파일 경로 가져오기
    csv_files = glob.glob("../data/raw/*.csv") # train.py 위치에 따라 경로 수정 필요 (예: "./data/raw/*.csv")
    
    if len(csv_files) == 0:
        print("⚠️ data/raw/ 폴더에 CSV 파일이 없습니다!")
        return

    all_X = []
    all_y = []

    print("=== 데이터 로딩 및 전처리 시작 ===")
    for file in csv_files:
        print(f"읽는 중: {file}")
        df = pd.read_csv(file)
        
        # 특징(X)과 정답(y) 추출
        features = df[['ear', 'mar', 'pitch']].values
        labels = df['label'].values
        
        # 주의: 파일별로 시퀀스를 따로 만들어야 시간 흐름이 꼬이지 않음
        X_seq, y_seq = create_sequences(features, labels, SEQ_LENGTH)
        
        if len(X_seq) > 0:
            all_X.append(X_seq)
            all_y.append(y_seq)

    # 리스트에 담긴 배열들을 하나의 거대한 배열로 통합
    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    print(f"총 생성된 시퀀스 묶음(데이터셋) 개수: {len(X_final)}개")

    # PyTorch 텐서로 변환
    X_tensor = torch.tensor(X_final, dtype=torch.float32)
    y_tensor = torch.tensor(y_final, dtype=torch.float32).view(-1, 1)

    # 모델에 넣기 좋게 배치(Batch) 단위로 쪼개주는 DataLoader 세팅
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 4. 학습 시작 ---
    model = SleepLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    criterion = nn.BCELoss() # 이진 분류용 오차 함수
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n=== AI 모델 학습 시작 ===")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()      # 이전 기울기 초기화
            outputs = model(batch_x)   # 모델 예측
            loss = criterion(outputs, batch_y) # 정답과 비교하여 오차(Loss) 계산
            loss.backward()            # 오차 역전파
            optimizer.step()           # 가중치 업데이트
            
            epoch_loss += loss.item()
            
        # 10 에포크마다 진행 상황 출력
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss/len(dataloader):.4f}')

    # --- 5. 가중치 저장 ---
    save_dir = '../data/weights'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'lstm_sleep_model.pth')
    
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 학습 완료! 뇌(가중치) 파일이 저장되었습니다: {save_path}")

if __name__ == '__main__':
    train_model()