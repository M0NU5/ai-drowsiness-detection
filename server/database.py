from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from config import DATABASE_URL
import datetime

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SleepLog(Base):
    __tablename__ = "sleep_logs"

    id = Column(Integer, primary_key=True, index=True)
    sleep_start_time = Column(DateTime, default=datetime.datetime.utcnow)
    sleep_end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    reason = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)