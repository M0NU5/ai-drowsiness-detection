from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from socket_handler import sio
import socketio

app = FastAPI()

# Socket.io 앱 마운트
socket_app = socketio.ASGIApp(sio, app)

# 클라이언트 정적 파일 서빙 (CORS 문제 방지용)
# 브라우저에서 http://localhost:8000/ 으로 접속 가능
app.mount("/", StaticFiles(directory="../client", html=True), name="client")

# 실행: uvicorn app:socket_app --reload