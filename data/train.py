import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# --- 1. 하이퍼파라미터 설정 (소량 데이터 & 과적합 방지 버전) ---
SEQ_LENGTH = 30     # 클라이언트와 동일하게 30 유지
INPUT_SIZE = 3      # EAR, MAR, Pitch
HIDDEN_SIZE = 128   # 뇌 용량 2배 유지
NUM_LAYERS = 3      # 신경망 깊이 유지
BATCH_SIZE = 16     # (수정) 데이터를 더 잘게 쪼개서 자주 가중치 업데이트
NUM_EPOCHS = 300    # (수정) 학습 반복 횟수 대폭 증가
LEARNING_RATE = 0.001

# --- 2. 모델 정의 (LSTM) ---
class SleepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SleepLSTM, self).__init__()
        # 🔥 (수정) dropout=0.5 추가: 학습 시 신경망의 50%를 무작위로 꺼서 특정 데이터 암기(과적합)를 방지
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.5
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return self.sigmoid(out)

# --- 3. 데이터 준비 함수 (Sliding Window) ---
def create_sequences(data, labels, seq_length):
    xs, ys = [], []
    if len(data) <= seq_length:
        return np.array(xs), np.array(ys)
        
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = labels[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model():
    # 🔥 GPU(CUDA) 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 현재 학습 장치: {device} ===")

    # 프로젝트 루트 기준 경로 설정
    csv_files = glob.glob("raw/*.csv") 
    
    if len(csv_files) == 0:
        print("⚠️ data/raw/ 폴더에 CSV 파일이 없습니다!")
        return

    all_X = []
    all_y = []

    print("\n=== 데이터 로딩 및 전처리 시작 ===")
    for file in csv_files:
        print(f"읽는 중: {file}")
        df = pd.read_csv(file)
        
        # 특징(X)과 정답(y) 추출
        features = df[['ear', 'mar', 'pitch']].values
        labels = df['label'].values
        
        X_seq, y_seq = create_sequences(features, labels, SEQ_LENGTH)
        
        if len(X_seq) > 0:
            all_X.append(X_seq)
            all_y.append(y_seq)

    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    print(f"총 생성된 시퀀스 데이터 개수: {len(X_final)}개")

    # PyTorch 텐서 변환
    X_tensor = torch.tensor(X_final, dtype=torch.float32)
    y_tensor = torch.tensor(y_final, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 모델을 생성하고 GPU로 전송
    model = SleepLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 최고 성능 모델 저장을 위한 변수
    best_loss = float('inf')
    save_dir = 'data/weights'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'lstm_sleep_model.pth')

    print("\n=== AI 모델 학습 시작 ===")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        
        model.train()
        for batch_x, batch_y in dataloader:
            # 🔥 데이터를 GPU로 전송
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        
        # 10 에포크마다 출력 및 최고 성능 갱신 시 저장
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

    print(f"\n✅ 학습 완료! 가장 오차가 적은 모델이 저장되었습니다: {save_path}")
    print(f"👑 최종 달성한 최저 Loss: {best_loss:.4f}")

if __name__ == '__main__':
    train_model()