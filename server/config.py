import os
from dotenv import load_dotenv

# .env 파일 로드 (루트 경로 기준)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# 데이터베이스
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/sleepdb")

# 서버
PORT = int(os.getenv("PORT", 8000))

# AI 모델
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 30))
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "../data/weights/lstm_sleep_model.pth")
MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", 3))
MODEL_HIDDEN_SIZE = int(os.getenv("MODEL_HIDDEN_SIZE", 64))
MODEL_NUM_LAYERS = int(os.getenv("MODEL_NUM_LAYERS", 2))