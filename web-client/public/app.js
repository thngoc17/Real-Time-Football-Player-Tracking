// File: /web-client/public/app.js

// --- CÁC BIẾN TRẠNG THÁI (STATE) TỔNG THỂ ---
const video = document.getElementById('sourceVideo');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d', { alpha: false });

const btnPlay = document.getElementById('togglePlay');
const fileUpload = document.getElementById('videoUpload');
const toggleBBox = document.getElementById('toggleBBox');
const statusUI = document.getElementById('apiStatus');

let isPlaying = false;
let animationFrameId;
let currentBoundingBoxes = [];
let globalScaleFactor = 1;

// --- CẤU HÌNH KIẾN TRÚC HYBRID (BUFFER 10% + REAL-TIME) ---
let isApiProcessing = false;
let bufferedResults = [];
let isBuffered = false;
let bufferTargetDuration = 0;

const INFERENCE_FPS_LIMIT = 5;
const INFERENCE_INTERVAL = 1000 / INFERENCE_FPS_LIMIT;
let lastInferenceTime = 0;

let frameCount = 0;
let lastFpsTime = performance.now();

// --- 1. LẮNG NGHE SỰ KIỆN NẠP VIDEO ---
fileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Reset toàn bộ state khi nạp video mới
    isBuffered = false;
    bufferedResults = [];
    currentBoundingBoxes = [];
    btnPlay.innerText = "Play / Pause";

    const fileURL = URL.createObjectURL(file);
    video.src = fileURL;

    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // YÊU CẦU 1: Đặt mốc thời lượng 10% của video
        bufferTargetDuration = video.duration * 0.1;

        btnPlay.disabled = false;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    };
});

// --- 2. QUẢN LÝ LUỒNG ĐIỀU KHIỂN & PRE-FETCH ---
btnPlay.addEventListener('click', async () => {
    if (video.paused) {
        // Nếu chưa pre-fetch 10% data -> Khởi chạy tiến trình Buffering ngầm
        if (!isBuffered) {
            btnPlay.disabled = true;
            btnPlay.innerText = "Buffering 10%...";
            await buildInitialBuffer();
            isBuffered = true;
            btnPlay.disabled = false;
            btnPlay.innerText = "Play / Pause";
        }

        // Sau khi hoàn tất 10%, tiến hành render video đồng thời
        video.play();
        isPlaying = true;
        renderLoop();
    } else {
        video.pause();
        isPlaying = false;
        cancelAnimationFrame(animationFrameId);
    }
});

// Tiến trình ngầm: Tua video, chụp frame, gọi API và lưu trữ trước 10% thời lượng
async function buildInitialBuffer() {
    let tempTime = 0;
    statusUI.innerText = "Pre-fetching 10%...";
    statusUI.className = "status-syncing";

    while (tempTime <= bufferTargetDuration) {
        // Bắt buộc video tua đến thời gian tương ứng
        await new Promise(resolve => {
            video.onseeked = resolve;
            video.currentTime = tempTime;
        });

        // Vẽ frame hiện tại ra canvas để trích xuất Base64
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        try {
            const result = await fetchInferenceData();
            bufferedResults.push({
                time: tempTime,
                detections: result.detections || [],
                scale: globalScaleFactor
            });
        } catch (err) {
            console.error("Buffer error at timestamp", tempTime, err);
        }

        // Tịnh tiến khung hình theo tốc độ FPS giới hạn
        tempTime += (1 / INFERENCE_FPS_LIMIT);
    }

    // Reset video về 0 để chuẩn bị phát thực tế
    video.currentTime = 0;
    statusUI.innerText = "Buffer Ready";
    statusUI.className = "status-ok";
}

// Hàm Core: Giao tiếp Azure ML (Dùng chung cho cả Buffer và Real-time)
async function fetchInferenceData() {
    const MAX_WIDTH = 640;
    let base64Frame;

    if (canvas.width > MAX_WIDTH) {
        globalScaleFactor = canvas.width / MAX_WIDTH;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = MAX_WIDTH;
        tempCanvas.height = canvas.height / globalScaleFactor;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height);
        base64Frame = tempCanvas.toDataURL('image/jpeg', 0.6);
    } else {
        globalScaleFactor = 1;
        base64Frame = canvas.toDataURL('image/jpeg', 0.6);
    }

    const startTime = performance.now();
    const response = await fetch('/api/proxy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Frame })
    });

    if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
    const result = await response.json();
    const endTime = performance.now();

    document.getElementById('latency').innerText = Math.round(endTime - startTime);
    document.getElementById('inference').innerText = (result.telemetry && result.telemetry.inference_ms) || 0;

    return result;
}

// --- 3. LUỒNG ĐỒ HỌA (RENDER LOOP) ---
function renderLoop() {
    if (!isPlaying) return;
    const currentTime = performance.now();

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // KIẾN TRÚC HYBRID: Quyết định dùng Buffer hay gọi API trực tiếp
    if (video.currentTime <= bufferTargetDuration && bufferedResults.length > 0) {
        // Giai đoạn 10% đầu: Tìm JSON có timestamp gần nhất trong Buffer
        const closest = bufferedResults.reduce((prev, curr) =>
            Math.abs(curr.time - video.currentTime) < Math.abs(prev.time - video.currentTime) ? curr : prev
        );
        currentBoundingBoxes = closest.detections;
        globalScaleFactor = closest.scale;
    } else {
        // Giai đoạn 90% sau: Gọi API Real-time (Throttling)
        if (currentTime - lastInferenceTime >= INFERENCE_INTERVAL) {
            lastInferenceTime = currentTime;
            executeRealtimeInference();
        }
    }

    // Vẽ hộp giới hạn
    if (toggleBBox.checked && currentBoundingBoxes.length > 0) {
        drawBoundingBoxes(currentBoundingBoxes);
    }

    // Đo lường Telemetry FPS
    frameCount++;
    if (currentTime - lastFpsTime >= 1000) {
        document.getElementById('fps').innerText = frameCount;
        frameCount = 0;
        lastFpsTime = currentTime;
    }

    animationFrameId = requestAnimationFrame(renderLoop);
}

// Giao tiếp bất đồng bộ cho 90% thời lượng sau của video
async function executeRealtimeInference() {
    if (isApiProcessing) return;
    isApiProcessing = true;
    statusUI.innerText = "Syncing...";
    statusUI.className = "status-syncing";

    try {
        const result = await fetchInferenceData();
        currentBoundingBoxes = result.detections || [];
        statusUI.innerText = "Connected";
        statusUI.className = "status-ok";
    } catch (error) {
        console.error("Realtime Inference Error:", error);
        statusUI.innerText = "Error / Timeout";
        statusUI.className = "status-error";
    } finally {
        isApiProcessing = false;
    }
}

// --- 4. YÊU CẦU 2: LOGIC VẼ BOUNDING BOX THEO INFERENCE.PY ---
function drawBoundingBoxes(detections) {
    detections.forEach(det => {
        const x1 = det.bbox[0] * globalScaleFactor;
        const y1 = det.bbox[1] * globalScaleFactor;
        const x2 = det.bbox[2] * globalScaleFactor;
        const y2 = det.bbox[3] * globalScaleFactor;
        const width = x2 - x1;
        const height = y2 - y1;

        const className = det.class_name;
        const confidence = det.confidence;

        // Mã gốc cv2.rectangle: BGR (0, 255, 0) -> RGB là Green (#00ff00)
        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);

        ctx.font = "14px Arial";

        if (className.toLowerCase() === 'player') {
            let yOffset = 0;

            // Phân loại Màu áo: BGR (0, 0, 255) -> RGB là Red (#ff0000)
            if (det.color_class !== null && det.color_class !== undefined) {
                const colorConf = det.color_conf !== undefined ? det.color_conf : 0;
                const colorLabel = `Color: ${det.color_class} (${colorConf.toFixed(2)})`;

                ctx.fillStyle = "#ff0000";
                // Ràng buộc tọa độ Y max bằng chiều cao canvas - 5
                const textY = Math.min(canvas.height - 5, y2 + 20 + yOffset);
                ctx.fillText(colorLabel, x1, textY);
                yOffset += 20;
            }

            // Phân loại Số áo: BGR (255, 0, 0) -> RGB là Blue (#0000ff)
            if (det.vis_class !== null && det.vis_class !== undefined) {
                const visConf = det.vis_conf !== undefined ? det.vis_conf : 0;
                const visLabel = `Number: ${det.vis_class} (${visConf.toFixed(2)})`;

                ctx.fillStyle = "#0000ff";
                // Ràng buộc tọa độ Y max bằng chiều cao canvas - 5
                const textY = Math.min(canvas.height - 5, y2 + 20 + yOffset);
                ctx.fillText(visLabel, x1, textY);
            }
        } else {
            // Label mặc định: BGR (0, 255, 0) -> RGB là Green (#00ff00)
            const label = `${className} ${confidence.toFixed(2)}`;
            ctx.fillStyle = "#00ff00";
            // Ràng buộc tọa độ Y min là 15 để không bị khuất khỏi màn hình
            const textY = Math.max(15, y1 - 10);
            ctx.fillText(label, x1, textY);
        }
    });
}