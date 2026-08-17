// File: /web-client/public/app.js

// --- CÁC BIẾN TRẠNG THÁI (STATE) ---
const video = document.getElementById('sourceVideo');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d', { alpha: false }); // Tối ưu phần cứng: vô hiệu hóa kênh alpha

// UI Elements
const btnPlay = document.getElementById('togglePlay');
const fileUpload = document.getElementById('videoUpload');
const toggleBBox = document.getElementById('toggleBBox');
const statusUI = document.getElementById('apiStatus');

// Cấu hình luồng
let isPlaying = false;
let animationFrameId;
let currentBoundingBoxes = [];
let isApiProcessing = false;

// Tối ưu Delay: Tỉ lệ scale để vẽ bounding box khớp với kích thước thật nếu ảnh bị thu nhỏ
let globalScaleFactor = 1;

// Cấu hình Throttling (Giới hạn gọi API)
const INFERENCE_FPS_LIMIT = 5; // Chỉ gọi Azure 5 lần/giây để tiết kiệm băng thông
const INFERENCE_INTERVAL = 1000 / INFERENCE_FPS_LIMIT;
let lastInferenceTime = 0;

// Các biến đo lường Telemetry
let frameCount = 0;
let lastFpsTime = performance.now();

// --- LẮNG NGHE SỰ KIỆN NẠP VIDEO ---
fileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const fileURL = URL.createObjectURL(file);
    video.src = fileURL;

    // Khi video đã load xong metadata, khởi tạo canvas
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        btnPlay.disabled = false;

        // Vẽ frame đầu tiên lên canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    };
});

// --- SỰ KIỆN PLAY/PAUSE ---
btnPlay.addEventListener('click', () => {
    if (video.paused) {
        video.play();
        isPlaying = true;
        renderLoop(); // Khởi động luồng đồ họa
    } else {
        video.pause();
        isPlaying = false;
        cancelAnimationFrame(animationFrameId); // Dừng luồng đồ họa
    }
});

// --- LUỒNG 1: RENDER ĐỒ HỌA (Chạy 30-60 FPS) ---
function renderLoop() {
    if (!isPlaying) return;

    const currentTime = performance.now();

    // 1. Vẽ video gốc lên canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 2. Vẽ Bounding Boxes (đè lên video)
    if (toggleBBox.checked && currentBoundingBoxes.length > 0) {
        drawBoundingBoxes(currentBoundingBoxes);
    }

    // 3. Tính toán Client FPS
    frameCount++;
    if (currentTime - lastFpsTime >= 1000) {
        document.getElementById('fps').innerText = frameCount;
        frameCount = 0;
        lastFpsTime = currentTime;
    }

    // 4. Kích hoạt luồng Inference nếu đủ thời gian (Throttling)
    if (currentTime - lastInferenceTime >= INFERENCE_INTERVAL) {
        lastInferenceTime = currentTime;
        executeInference();
    }

    // Tiếp tục vòng lặp
    animationFrameId = requestAnimationFrame(renderLoop);
}

// --- LUỒNG 2: GIAO TIẾP API BẤT ĐỒNG BỘ ---
async function executeInference() {
    // Ngăn chặn việc bắn request mới nếu request cũ chưa phản hồi
    if (isApiProcessing) return;

    isApiProcessing = true;
    statusUI.innerText = "Syncing...";
    statusUI.className = "status-syncing";

    try {
        // TỐI ƯU DELAY #1: Resize ảnh xuống tối đa 640px trước khi gửi
        const MAX_WIDTH = 640;
        let base64Frame;

        if (canvas.width > MAX_WIDTH) {
            globalScaleFactor = canvas.width / MAX_WIDTH;

            // Tạo canvas tạm để thu nhỏ ảnh
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = MAX_WIDTH;
            tempCanvas.height = canvas.height / globalScaleFactor;

            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height);

            // Chỉ số 0.6 giảm dung lượng ảnh đi khoảng 60% so với gốc
            base64Frame = tempCanvas.toDataURL('image/jpeg', 0.6);
        } else {
            globalScaleFactor = 1;
            base64Frame = canvas.toDataURL('image/jpeg', 0.6);
        }

        const startTime = performance.now();

        // Gửi qua Vercel Serverless Proxy
        const response = await fetch('/api/proxy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Frame })
        });

        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

        const result = await response.json();
        const endTime = performance.now();

        // SỬA LỖI #2 & #3: Lấy đúng key detections từ response đã được flatten
        currentBoundingBoxes = result.detections || [];

        // Cập nhật Telemetry
        document.getElementById('latency').innerText = Math.round(endTime - startTime);
        // Lấy inference_ms thông qua object telemetry đã set ở proxy.js
        document.getElementById('inference').innerText = (result.telemetry && result.telemetry.inference_ms) || 0;

        statusUI.innerText = "Connected";
        statusUI.className = "status-ok";

    } catch (error) {
        console.error("Inference Error:", error);
        statusUI.innerText = "Error / Timeout";
        statusUI.className = "status-error";
    } finally {
        isApiProcessing = false;
    }
}

// --- HÀM VẼ BOUNDING BOX ---
// SỬA LỖI #2: Đã xóa dòng "currentBoundingBoxes = result.detections || [];" rác bên ngoài hàm

function drawBoundingBoxes(detections) {
    detections.forEach(det => {
        // Áp dụng globalScaleFactor để đưa tọa độ về đúng kích thước video gốc
        const x = det.bbox[0] * globalScaleFactor;
        const y = det.bbox[1] * globalScaleFactor;
        const width = (det.bbox[2] - det.bbox[0]) * globalScaleFactor;
        const height = (det.bbox[3] - det.bbox[1]) * globalScaleFactor;

        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        ctx.fillStyle = "#00ff00";
        ctx.font = "14px Arial";

        // Cập nhật key theo API mới
        let label = `${det.class_name} ${Math.round(det.confidence * 100)}%`;
        if (det.color_class) label += ` | ${det.color_class}`;

        ctx.fillText(label, x, y - 5);
    });
}