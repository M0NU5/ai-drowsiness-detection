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
let alarmInterval = null; // 5초 반복 타이머를 저장할 변수
let isRunning = false;
let isCurrentlySleeping = false;
// 두 점 사이의 유클리디안 거리 계산 함수
function calcDistance(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}
function playAlarm() {
    alarmAudio.currentTime = 0; // 재생할 때마다 소리를 맨 처음으로 되감기
    let playPromise = alarmAudio.play();
    if (playPromise !== undefined) {
        playPromise.catch(error => console.log("오디오 에러 무시:", error));
    }
}
function startAlarmLoop() {
    // 수면 상태가 아니면(주의/정상) 타이머 사이클을 즉시 멈춤
    if (!isCurrentlySleeping) return; 
    
    console.log("알람 재생 (다음 재생은 5초 뒤)");
    alarmAudio.currentTime = 0; // 소리를 맨 앞으로 되감기
    let playPromise = alarmAudio.play();
    if (playPromise !== undefined) {
        playPromise.catch(e => console.log("무시됨"));
    }

    // 정확히 5초(5000ms) 뒤에 자기 자신을 다시 부름 (절대 끊기지 않음)
    alarmInterval = setTimeout(startAlarmLoop, 5000);
}
// EAR (눈 깜빡임 비율) 계산
function getEAR(landmarks, leftIndices) {
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

// 🔥 추가됨: MAR (입 벌림) 계산
function getMAR(landmarks) {
    const topLip = landmarks[13];
    const bottomLip = landmarks[14];
    const leftCorner = landmarks[78];
    const rightCorner = landmarks[308];

    const vertical = calcDistance(topLip, bottomLip);
    const horizontal = calcDistance(leftCorner, rightCorner);
    return vertical / horizontal;
}

// 🔥 추가됨: Pitch (고개 각도) 계산
function getPitch(landmarks) {
    const topHead = landmarks[10];
    const chin = landmarks[152];
    const noseTip = landmarks[1];
    
    const faceHeight = calcDistance(topHead, chin);
    const noseToChin = calcDistance(noseTip, chin);
    return noseToChin / faceHeight;
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
        
        const mar = getMAR(landmarks);
        const pitch = getPitch(landmarks);

        // 마커 그리기 (시각화)
        ctx.fillStyle = 'lime';
        landmarks.forEach(p => {
            ctx.fillRect(p.x * canvas.width, p.y * canvas.height, 2, 2);
        });

        // 서버로 특징 데이터 전송 (AI 판단용 3가지 필수 데이터 모두 포함)
        socket.emit('process_frame', {
            face_detected: true,
            ear: avgEAR,
            mar: mar,
            pitch: pitch,
            timestamp: Date.now()
        });
    } else {
        // 얼굴을 놓쳤을 경우 예외 처리
        socket.emit('process_frame', { face_detected: false, ear: 0, mar: 0, pitch: 0, timestamp: Date.now() });
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
socket.on('sleep_alert', (data) => {
    
    if (data.status === 'sleep') {
        statusText.innerText = "상태: 수면 감지됨! (알람 발생)";
        canvas.classList.add('alert');
        
        // 이미 알람 사이클이 돌고 있다면 무시 (소리 겹침 방지)
        if (isCurrentlySleeping) return; 
        isCurrentlySleeping = true;
        
        // 수면 상태 진입 즉시 5초 반복 사이클 작동 시작!
        startAlarmLoop();
        
    } else if (data.status === 'warning') {
        statusText.innerText = "상태: 졸음 주의 (경계 단계)";
        canvas.classList.remove('alert');
        
        // 주의 단계일 때 알람 끄고 사이클 파괴
        isCurrentlySleeping = false;
        if (alarmInterval) clearTimeout(alarmInterval);
        alarmAudio.pause();
        
    } else {
        statusText.innerText = "상태: 정상";
        canvas.classList.remove('alert');
        
        // 정상 상태일 때 알람 끄고 사이클 파괴 및 소리 되감기
        isCurrentlySleeping = false;
        if (alarmInterval) clearTimeout(alarmInterval);
        alarmAudio.pause();
        alarmAudio.currentTime = 0;
    }
});