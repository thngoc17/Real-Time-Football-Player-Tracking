import os
import json
import base64
import numpy as np
import cv2
from ultralytics import YOLO
from inference import JerseyClassifier

# Khai báo global ở cấp độ script để giữ model trên RAM của worker
detector = None
classifier = None

def init():
    """Khởi tạo mô hình một lần khi container khởi động."""
    global detector, classifier
    
    # Biến môi trường này được Azure ML tự động cung cấp
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    
    # Đường dẫn thư mục phải khớp chính xác với tên đăng ký lúc chạy `az ml model create`
    yolo_path = os.path.join(model_dir, "models", "best.pt")
    color_path = os.path.join(model_dir, "models", "jersey_color_model.pth")
    vis_path = os.path.join(model_dir, "models", "jersey_visibility_model.pth")
    
    detector = YOLO(yolo_path)
    classifier = JerseyClassifier(color_model_path=color_path, visibility_model_path=vis_path)

def run(raw_data):
    """Thực thi suy luận trên từng request JSON."""
    try:
        data = json.loads(raw_data)
        img_bytes = base64.b64decode(data["image"])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Invalid image payload"}

        # Suy luận YOLO
        detection_results = detector(frame, conf=0.3, verbose=False)
        frame_detections = []
        crops_data = []
        
        for result in detection_results:
            if result.boxes is None: continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = detector.names[cls_id]

                if conf > 0.4 and x2 > x1 and y2 > y1:
                    det_info = {
                        "bbox": [x1, y1, x2, y2], "class_name": cls_name, "confidence": conf,
                        "color_class": None, "vis_class": None
                    }
                    frame_detections.append(det_info)
                    
                    if cls_name.lower() == 'player':
                        crop = frame[y1:y2, x1:x2]
                        crops_data.append(crop if crop.size > 0 else None)
        
        # Suy luận ResNet
        if crops_data:
            clf_results = classifier.classify_batch([c for c in crops_data if c is not None])
            crop_idx = 0
            for det in frame_detections:
                if det["class_name"].lower() == 'player' and crop_idx < len(clf_results):
                    c_cls, _, v_cls, _ = clf_results[crop_idx]
                    det["color_class"] = c_cls
                    det["vis_class"] = v_cls
                    crop_idx += 1
                    
        return {"status": "success", "detections": frame_detections}
        
    except Exception as e:
        return {"error": str(e)}