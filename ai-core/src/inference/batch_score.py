import os
import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from inference import JerseyClassifier, box_iou

detector = None
classifier = None

def init():
    """Khởi tạo mô hình tương tự Online Endpoint."""
    global detector, classifier
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    yolo_path = os.path.join(model_dir, "models", "best.pt")
    color_path = os.path.join(model_dir, "models", "jersey_color_model.pth")
    vis_path = os.path.join(model_dir, "models", "jersey_visibility_model.pth")
    
    detector = YOLO(yolo_path)
    classifier = JerseyClassifier(color_model_path=color_path, visibility_model_path=vis_path)

def run(mini_batch):
    """Xử lý danh sách các tệp video."""
    batch_results = []
    
    for video_path in mini_batch:
        # BẮT BUỘC: Reset state tracking cho mỗi video mới
        tracked_detections = {}
        next_detection_id = 0
        video_output = {"video": os.path.basename(video_path), "frames": []}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            batch_results.append({"video": video_path, "error": "Could not open video"})
            continue
            
        frame_count = 0
        while True:
            ret, orig_frame = cap.read()
            if not ret: break
            frame_count += 1
            
            # Mô phỏng logic nhảy frame (skip frame) từ mã gốc
            if frame_count % 3 == 1:
                det_results = detector(orig_frame, conf=0.3, verbose=False)
                frame_detections = []
                crops_data = []
                
                for result in det_results:
                    if result.boxes is None: continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cls_name = detector.names[int(box.cls[0].cpu().numpy())]
                        
                        if conf > 0.4 and x2 > x1 and y2 > y1:
                            bbox = (x1, y1, x2, y2)
                            
                            # Logic Tracking
                            best_match_id = None
                            best_iou = 0.0
                            for det_id, prev_bbox in tracked_detections.items():
                                iou = box_iou(bbox, prev_bbox)
                                if iou > best_iou and iou > 0.3:
                                    best_iou = iou
                                    best_match_id = det_id
                                    
                            if best_match_id is None:
                                best_match_id = next_detection_id
                                next_detection_id += 1
                                
                            tracked_detections[best_match_id] = bbox
                            
                            det_info = {
                                "id": best_match_id, "bbox": bbox, "class": cls_name, "conf": conf,
                                "color_class": None, "vis_class": None
                            }
                            frame_detections.append(det_info)
                            
                            if cls_name.lower() == 'player':
                                crop = orig_frame[y1:y2, x1:x2]
                                crops_data.append(crop if crop.size > 0 else None)
                                
                if crops_data:
                    clf_results = classifier.classify_batch([c for c in crops_data if c is not None])
                    crop_idx = 0
                    for det in frame_detections:
                        if det["class"].lower() == 'player' and crop_idx < len(clf_results):
                            det["color_class"] = clf_results[crop_idx][0]
                            det["vis_class"] = clf_results[crop_idx][2]
                            crop_idx += 1
                            
                video_output["frames"].append({"frame_id": frame_count, "detections": frame_detections})
                
        cap.release()
        batch_results.append(video_output)
        
    return batch_results