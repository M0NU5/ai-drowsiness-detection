from database import SessionLocal, SleepLog

db = SessionLocal()
logs = db.query(SleepLog).all()

print(f"\n=== 총 {len(logs)}개의 수면 기록이 저장되어 있습니다 ===")
for log in logs:
    start_time = log.sleep_start_time.strftime("%Y-%m-%d %H:%M:%S") if log.sleep_start_time else "N/A"
    print(f"ID: {log.id} | 사유: {log.reason} | 시작: {start_time} | 지속: {log.duration_seconds}초")

db.close()