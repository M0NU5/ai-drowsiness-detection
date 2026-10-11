import socketio
from collections import deque
from database import SessionLocal, SleepLog
from ai_model import SleepDetectorModel
from config import WINDOW_SIZE
import datetime

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

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
    
    data_buffer.append(data)
    
    # 🔥 [수정 1] 데이터가 30개(WINDOW_SIZE)가 안 모였다면, 아예 AI 판단을 건너뜁니다.
    # 이렇게 해야 버퍼가 비워진 직후에 오작동하는 것을 막을 수 있습니다.
    if len(data_buffer) < WINDOW_SIZE:
        return
        
    is_sleeping = model.predict(list(data_buffer))
    
    db = SessionLocal()
    
    if is_sleeping and current_state == "awake":
        current_state = "sleep"
        # face_detected 값이 없을 때를 대비해 기본값 처리를 안전하게 해줍니다.
        reason = "얼굴 이탈" if not data.get("face_detected", True) else "눈감음(모델판단)"
        
        new_log = SleepLog(reason=reason)
        db.add(new_log)
        db.commit()
        db.refresh(new_log)
        current_log_id = new_log.id
        
        await sio.emit("sleep_alert", {"status": "sleep"}, room=sid)
        
    elif not is_sleeping and current_state == "sleep":
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
        
        # 🔥 [수정 2] 상태가 정상(awake)으로 돌아왔을 때 과거의 졸음 잔상 데이터를 날려버립니다.
        data_buffer.clear()

    db.close()

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    # 🔥 [수정 3] 클라이언트가 나가면 찌꺼기가 남지 않게 버퍼를 비워줍니다.
    data_buffer.clear()