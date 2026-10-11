const socket = io(ENV.NGROK_URL, { 
    transports: ['websocket'], 
    extraHeaders: {
        "ngrok-skip-browser-warning": "true"
    }
});

const video = document.getElementById('videoElement');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const alarmAudio = document.getElementById('alarmAudio');
const statusText = document.getElementById('statusText');

let isRunning = false;

// 두 점 사이의 유클리디안 거리 계산 함수
function calcDistance(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

// EAR (눈 깜빡임 비율) 계산
function getEAR(landmarks, leftIndices) {
    // Left eye indices: 33, 160, 158, 133, 153, 144
    const p1 = landmarks[leftIndices[0]];
    const p2 = landmarks[leftIndices[1]];
    const p3 = landmarks[leftIndices[2]];
    const p4 = landmarks[leftIndices[3]];
    const p5 = landmarks[leftIndices[4]];
    const p6 = landmarks[leftIndices[5]];

    const vertical1 = calcDistance(p2, p6);
    const vertical2 = calcDistance(p3, p5);
    const horizontal = calcDistance(p1, p4);

    return (vertical1 + vertical2) / (2.0 * horizontal);
}

// MediaPipe Face Mesh 초기화
const faceMesh = new FaceMesh({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
}});

faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

faceMesh.onResults((results) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];
        
        // 왼쪽 눈 (33, 160, 158, 133, 153, 144), 오른쪽 눈 (362, 385, 387, 263, 373, 380)
        const leftEAR = getEAR(landmarks, [33, 160, 158, 133, 153, 144]);
        const rightEAR = getEAR(landmarks, [362, 385, 387, 263, 373, 380]);
        const avgEAR = (leftEAR + rightEAR) / 2.0;

        // 마커 그리기 (시각화)
        ctx.fillStyle = 'lime';
        landmarks.forEach(p => {
            ctx.fillRect(p.x * canvas.width, p.y * canvas.height, 2, 2);
        });

        // 서버로 특징 데이터 전송
        socket.emit('process_frame', {
            face_detected: true,
            ear: avgEAR,
            mar: 0.0, // MAR 및 Pitch 등 추가 구현 영역
            timestamp: Date.now()
        });
    } else {
        // 얼굴을 놓쳤을 경우 예외 처리
        socket.emit('process_frame', { face_detected: false, timestamp: Date.now() });
    }
});

// 카메라 세팅
const camera = new Camera(video, {
    onFrame: async () => {
        if (isRunning) await faceMesh.send({image: video});
    },
    width: 640,
    height: 480
});

// 시작 버튼: 오디오 Autoplay 정책 우회 및 모니터링 시작
startBtn.addEventListener('click', () => {
    // 빈 오디오 로드 및 재생/정지하여 사용자 인터랙션 등록
    alarmAudio.play().then(() => {
        alarmAudio.pause();
        alarmAudio.currentTime = 0;
    }).catch(e => console.log("Audio unlock failed", e));
    
    isRunning = true;
    camera.start();
    startBtn.disabled = true;
    startBtn.innerText = "모니터링 중...";
});

// 서버로부터 알람 수신
let currentUIState = 'awake';

socket.on('sleep_alert', (data) => {
    if (data.status !== currentUIState) {
        currentUIState = data.status;
        updateStatusUI(currentUIState);

        if (currentUIState === 'warning') {
            addLogItem("⚠️ 졸음 징후 감지 (주의 단계)", "var(--warning-color)");
        } else if (currentUIState === 'sleep') {
            addLogItem("🚨 수면 상태 진입 감지! (DB 기록 시작)", "var(--sleep-color)");
        } else if (currentUIState === 'awake') {
            addLogItem("✅ 정상 상태 복귀", "var(--awake-color)");
        }
    }
});

function updateStatusUI(status) {
    const badge = document.getElementById('statusBadge');
    const text = document.getElementById('statusText');
    const videoContainer = document.getElementById('videoContainer');
    
    // 기존 클래스 싹 비우기 (초기화)
    badge.className = 'status-badge';
    videoContainer.className = 'video-section';

    if (status === 'warning') {
        badge.classList.add('warning');
        text.innerText = "WARNING (DROWSY)";
        videoContainer.classList.add('warning-alert');
    } else if (status === 'sleep') {
        badge.classList.add('sleep');
        text.innerText = "SLEEPING DETECTED";
        videoContainer.classList.add('sleep-alert');
    } else {
        text.innerText = "AWAKE";
    }
}

// 로그 아이템 추가 시 색상을 인자로 받도록 수정
function addLogItem(message, color) {
    const logList = document.getElementById('logList');
    const now = new Date();
    const timeString = now.toTimeString().split(' ')[0];
    
    const logHtml = `
        <div class="log-item" style="border-left-color: ${color};">
            <div class="log-time">${timeString}</div>
            <div>${message}</div>
        </div>
    `;
    logList.insertAdjacentHTML('afterbegin', logHtml);
}
