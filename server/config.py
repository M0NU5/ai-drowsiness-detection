import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# .env 파일이 없거나 설정되지 않았을 때 더미/빈 값을 넣거나 안내 메시지 출력
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("⚠️ .env 파일에 DATABASE_URL 설정이 필요합니다!")

PORT = int(os.getenv("PORT", 8000))

# AI 모델
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 30))
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "../data/weights/lstm_sleep_model.pth")
MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", 3))
MODEL_HIDDEN_SIZE = int(os.getenv("MODEL_HIDDEN_SIZE", 128))
MODEL_NUM_LAYERS = int(os.getenv("MODEL_NUM_LAYERS", 3))