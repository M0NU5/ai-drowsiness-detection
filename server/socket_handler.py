import socketio
from collections import deque
from database import SessionLocal, SleepLog
from ai_model import SleepDetectorModel
from config import WINDOW_SIZE
import datetime

# FastAPI와 호환되는 Async 서버
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# 상태 관리 변수 (단일 유저 기준)
data_buffer = deque(maxlen=WINDOW_SIZE)
model = SleepDetectorModel()

current_state = "awake"
current_log_id = None

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def process_frame(sid, data):
    global current_state, current_log_id
    
    # 1. 큐에 데이터 삽입
    data_buffer.append(data)
    
    # 2. 모델 추론
    is_sleeping = model.predict(list(data_buffer))
    
    db = SessionLocal()
    
    # 3. 상태 변화 감지 및 DB 로깅
    if is_sleeping and current_state == "awake":
        # 정상 -> 수면 전환
        current_state = "sleep"
        reason = "얼굴 이탈" if not data.get("face_detected") else "눈감음(모델판단)"
        
        new_log = SleepLog(reason=reason)
        db.add(new_log)
        db.commit()
        db.refresh(new_log)
        current_log_id = new_log.id
        
        # 클라이언트에 알람 신호 전송
        await sio.emit("sleep_alert", {"status": "sleep"}, room=sid)
        
    elif not is_sleeping and current_state == "sleep":
        # 수면 -> 정상 전환 (기존 로그 업데이트)
        current_state = "awake"
        if current_log_id:
            log = db.query(SleepLog).filter(SleepLog.id == current_log_id).first()
            if log:
                log.sleep_end_time = datetime.datetime.utcnow()
                delta = log.sleep_end_time - log.sleep_start_time
                log.duration_seconds = int(delta.total_seconds())
                db.commit()
            current_log_id = None
            
        await sio.emit("sleep_alert", {"status": "awake"}, room=sid)

    db.close()

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")