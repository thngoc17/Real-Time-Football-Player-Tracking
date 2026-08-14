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
        // Nén frame hiện tại thành Base64 JPEG.
        // TỐI QUAN TRỌNG: Chỉ số 0.6 giảm dung lượng ảnh đi khoảng 60% so với gốc.
        const base64Frame = canvas.toDataURL('image/jpeg', 0.6);
        const startTime = performance.now();

        // Gửi qua Vercel Serverless Proxy (Ngày 4 của Sprint)
        const response = await fetch('/api/proxy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Frame })
        });

        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

        const result = await response.json();
        const endTime = performance.now();

        // Cập nhật trạng thái dùng để render
        currentBoundingBoxes = result.objects || [];

        // Cập nhật Telemetry
        document.getElementById('latency').innerText = Math.round(endTime - startTime);
        // Giả sử Azure ML trả về số liệu inference_time bên trong JSON
        document.getElementById('inference').innerText = result.inference_time_ms || 0;

        statusUI.innerText = "Connected";
        statusUI.className = "status-ok";

    } catch (error) {
        console.error("Inference Error:", error);
        statusUI.innerText = "Error / Timeout";
        statusUI.className = "status-error";
        // Graceful Degradation: Không xóa bounding box cũ ngay lập tức,
        // hoặc bạn có thể gán currentBoundingBoxes = [] tùy quy trình.
    } finally {
        isApiProcessing = false;
    }
}

// --- HÀM VẼ BOUNDING BOX ---
function drawBoundingBoxes(objects) {
    // Giả định chuẩn JSON: { objects: [ { class: "player", conf: 0.95, xmin: 10, ymin: 20, xmax: 50, ymax: 100 } ] }
    // Bạn cần điều chỉnh key cho khớp với code score.py của bạn.
    objects.forEach(obj => {
        const x = obj.xmin;
        const y = obj.ymin;
        const width = obj.xmax - obj.xmin;
        const height = obj.ymax - obj.ymin;

        // Cấu hình bút vẽ
        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Vẽ nhãn (Label)
        ctx.fillStyle = "#00ff00";
        ctx.font = "14px Arial";
        const label = `${obj.class} ${Math.round(obj.conf * 100)}%`;
        ctx.fillText(label, x, y - 5);
    });
}