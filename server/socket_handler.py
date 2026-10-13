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
sleep_counter = 0  #졸음 지속 시간을 체크할 카운터 추가

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def process_frame(sid, data):
    global current_state, current_log_id, sleep_counter
    
    data_buffer.append(data)
    if len(data_buffer) < WINDOW_SIZE:
        return
        
    is_sleeping = model.predict(list(data_buffer))
    db = SessionLocal()
    
    try:
        # 모델이 수면으로 판단하면 카운터를 올리고, 아니면 0으로 초기화
        if is_sleeping:
            sleep_counter += 1
        else:
            sleep_counter = 0

        # 상태 결정 (카운터 기준)
        new_state = "awake"
        if sleep_counter >= 25:    # 연속 수면 -> 완전 수면 (DB 저장)
            new_state = "sleep"
        elif sleep_counter >= 5:   # 약 0.2초 연속 수면 -> 주의 (경고만 줌)
            new_state = "warning"

        # 상태가 바뀌었을 때만 클라이언트로 알림 전송 및 DB 처리
        if new_state != current_state:
            
            # 1. 정상 -> 주의
            if new_state == "warning" and current_state == "awake":
                current_state = "warning"
                await sio.emit("sleep_alert", {"status": "warning"}, room=sid)

            # 2. 주의 -> 수면 (DB 저장 시작)
            elif new_state == "sleep" and current_state == "warning":
                current_state = "sleep"
                reason = "얼굴 이탈" if not data.get("face_detected", True) else "눈감음 지속"
                
                print("\n💤 수면 감지! DB에 새 기록 저장을 시도합니다...")
                new_log = SleepLog(reason=reason)
                db.add(new_log)
                db.commit()
                db.refresh(new_log)
                current_log_id = new_log.id
                print(f"✅ DB 저장 성공! 기록 ID: {current_log_id}\n")
                
                await sio.emit("sleep_alert", {"status": "sleep"}, room=sid)

            # 3. 수면/주의 -> 정상 기상 (DB 업데이트)
            elif new_state == "awake":
                if current_state == "sleep" and current_log_id:
                    print(f"\n⏰ 기상 감지! DB 업데이트(종료시간 기록)를 시도합니다...")
                    log = db.query(SleepLog).filter(SleepLog.id == current_log_id).first()
                    if log:
                        log.sleep_end_time = datetime.datetime.utcnow()
                        delta = log.sleep_end_time - log.sleep_start_time
                        log.duration_seconds = int(delta.total_seconds())
                        db.commit()
                        print("✅ DB 업데이트 성공!\n")
                    current_log_id = None
                    
                current_state = "awake"
                await sio.emit("sleep_alert", {"status": "awake"}, room=sid)
                data_buffer.clear()

    except Exception as e:
        db.rollback()
        print(f"\n [DB 작업 실패 원인]: {e}\n")
    finally:
        db.close()

@sio.event
async def disconnect(sid):
    global sleep_counter
    sleep_counter = 0
    data_buffer.clear()
    print(f"Client disconnected: {sid}")