import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from socket_handler import sio
import socketio
from config import PORT

app = FastAPI()

socket_app = socketio.ASGIApp(sio, app)

# 클라이언트 폴더 마운트
app.mount("/", StaticFiles(directory="../client", html=True), name="client")

if __name__ == "__main__":
    # app.py를 직접 실행할 경우 .env의 PORT를 사용합니다.
    uvicorn.run("app:socket_app", host="0.0.0.0", port=PORT, reload=True)