import os

# PostgreSQL 연결 정보 (본인 환경에 맞게 수정)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/sleepdb")
WINDOW_SIZE = 30 # 시계열 추론을 위한 버퍼 사이즈 (프레임 수)