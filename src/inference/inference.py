import cv2
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import numpy as np
import yaml
import sys

# Optimized preprocessing for ResNet models
preprocess = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# NOTE: detector and jersey_classifier will be initialized in __main__ after reading inference.yaml
detector = None


class JerseyClassifier:
    """Combined jersey color and number visibility classifier"""

    def __init__(self, color_model_path, visibility_model_path):
        self.device = device

        # Load jersey color model
        self.color_model = None
        self.color_classes = []
        if color_model_path and Path(color_model_path).exists():
            try:
                color_checkpoint = torch.load(color_model_path, map_location=device)
                num_color_classes = len(color_checkpoint['class_names'])
                # Use a ResNet50 backbone as in training script; change if different
                self.color_model = models.resnet50(pretrained=False)
                self.color_model.fc = nn.Linear(self.color_model.fc.in_features, num_color_classes)
                self.color_model.load_state_dict(color_checkpoint['model_state_dict'])
                self.color_model = self.color_model.to(device)
                self.color_model.eval()
                self.color_classes = color_checkpoint['class_names']
                print(f"Loaded jersey color model: {num_color_classes} classes from {color_model_path}")

                if device.type == 'cuda':
                    # Optionally script for speed; keep original model for flexibility if JIT fails
                    try:
                        self.color_model = torch.jit.script(self.color_model)
                    except Exception as e:
                        print(f"Warning: torch.jit.script failed for color_model: {e}")

            except Exception as e:
                print(f"Error loading color model '{color_model_path}': {e}")
                self.color_model = None

        # Load number visibility model
        self.visibility_model = None
        self.visibility_classes = []
        if visibility_model_path and Path(visibility_model_path).exists():
            try:
                vis_checkpoint = torch.load(visibility_model_path, map_location=device)
                num_vis_classes = len(vis_checkpoint['class_names'])
                self.visibility_model = models.resnet50(pretrained=False)
                self.visibility_model.fc = nn.Linear(self.visibility_model.fc.in_features, num_vis_classes)
                self.visibility_model.load_state_dict(vis_checkpoint['model_state_dict'])
                self.visibility_model = self.visibility_model.to(device)
                self.visibility_model.eval()
                self.visibility_classes = vis_checkpoint['class_names']
                print(f"Loaded visibility model: {num_vis_classes} classes from {visibility_model_path}")

                if device.type == 'cuda':
                    try:
                        self.visibility_model = torch.jit.script(self.visibility_model)
                    except Exception as e:
                        print(f"Warning: torch.jit.script failed for visibility_model: {e}")

            except Exception as e:
                print(f"Error loading visibility model '{visibility_model_path}': {e}")
                self.visibility_model = None

    def classify_batch(self, crop_images):
        """Classify batch of crop images for both color and visibility"""
        if not crop_images:
            return []

        try:
            batch_tensors = []
            valid_indices = []

            for i, crop_image in enumerate(crop_images):
                if crop_image is None or crop_image.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)
                input_tensor = preprocess(pil_image)
                batch_tensors.append(input_tensor)
                valid_indices.append(i)

            if not batch_tensors:
                return [(None, 0.0, None, 0.0)] * len(crop_images)

            input_batch = torch.stack(batch_tensors).to(device)
            results = [(None, 0.0, None, 0.0)] * len(crop_images)

            with torch.no_grad():
                color_top_probs = color_top_indices = None
                if self.color_model is not None:
                    color_outputs = self.color_model(input_batch)
                    color_probs = torch.nn.functional.softmax(color_outputs, dim=1)
                    color_top_probs, color_top_indices = torch.topk(color_probs, 1, dim=1)

                vis_top_probs = vis_top_indices = None
                if self.visibility_model is not None:
                    vis_outputs = self.visibility_model(input_batch)
                    vis_probs = torch.nn.functional.softmax(vis_outputs, dim=1)
                    vis_top_probs, vis_top_indices = torch.topk(vis_probs, 1, dim=1)

                for i, valid_idx in enumerate(valid_indices):
                    color_class = None
                    color_conf = 0.0
                    if self.color_model is not None and color_top_indices is not None:
                        color_class = self.color_classes[color_top_indices[i].item()]
                        color_conf = color_top_probs[i].item()

                    visibility_class = None
                    visibility_conf = 0.0
                    if self.visibility_model is not None and vis_top_indices is not None:
                        visibility_class = self.visibility_classes[vis_top_indices[i].item()]
                        visibility_conf = vis_top_probs[i].item()

                    results[valid_idx] = (color_class, color_conf, visibility_class, visibility_conf)

            return results

        except Exception as e:
            print(f"Batch classification error: {e}")
            return [(None, 0.0, None, 0.0)] * len(crop_images)


def box_iou(boxA, boxB):
    """Compute IoU between two boxes (x1,y1,x2,y2)"""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0.0


def process_video_offline(video_path, output_path=None):
    """Process entire video offline, then play results"""
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"Error: Video file '{video_path}' does not exist!")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video")
        return False

    # Get video properties
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"Video: {total_frames} frames at {original_fps:.2f} FPS, resolution {original_w}x{original_h}")
    print("Starting offline processing...")

    # Setup output path
    if output_path is None:
        try:
            code_dir = Path(__file__).resolve().parent
        except NameError:
            code_dir = Path.cwd()
        outputs_dir = code_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = outputs_dir / (video_path.stem + "_processed.mp4")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, original_fps, (original_w, original_h))

    # Store all processed frames
    processed_frames = []
    frame_count = 0
    start_time = time.time()

    # Simple tracking
    tracked_detections = {}
    next_detection_id = 0

    print("Processing frames...")
    while True:
        ret, orig_frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process frame
        proc_frame = orig_frame
        scale_x = scale_y = 1.0

        # Resize if needed
        proc_h, proc_w = proc_frame.shape[:2]
        if proc_w > 1280:
            new_w = 1280
            new_h = int(proc_h * new_w / proc_w)
            proc_frame = cv2.resize(orig_frame, (new_w, new_h))
            proc_h, proc_w = proc_frame.shape[:2]
            scale_x = original_w / proc_w
            scale_y = original_h / proc_h

        annotated_frame = orig_frame.copy()

        # Run YOLO every 3 frames
        if frame_count % 3 == 1:
            # Run YOLO detection
            # detector must be initialized in main()
            global detector
            if detector is None:
                print("Error: detector is not initialized. Exiting.")
                cap.release()
                return False

            detection_results = detector(proc_frame, conf=0.3, verbose=False)

            crops_data = []
            frame_detections = []

            for result in detection_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1_p, y1_p, x2_p, y2_p = box.xyxy[0].cpu().numpy()

                        # Scale coordinates
                        x1 = max(0, min(int(x1_p * scale_x), original_w - 1))
                        y1 = max(0, min(int(y1_p * scale_y), original_h - 1))
                        x2 = max(x1 + 1, min(int(x2_p * scale_x), original_w))
                        y2 = max(y1 + 1, min(int(y2_p * scale_y), original_h))

                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = detector.names[class_id] if class_id < len(detector.names) else str(class_id)

                        if confidence > 0.4 and x2 > x1 and y2 > y1:
                            bbox = (x1, y1, x2, y2)

                            # Simple tracking - match with previous detections
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

                            detection_info = {
                                'bbox': bbox,
                                'class_name': class_name,
                                'confidence': confidence,
                                'id': best_match_id,
                                'color_class': None,
                                'color_conf': 0.0,
                                'vis_class': None,
                                'vis_conf': 0.0
                            }
                            frame_detections.append(detection_info)

                            # Prepare crops for classification
                            if class_name.lower() == 'player':
                                crop_image = orig_frame[y1:y2, x1:x2]
                                if (crop_image is not None and crop_image.size > 0 and
                                        crop_image.shape[0] > 20 and crop_image.shape[1] > 20):
                                    crops_data.append(crop_image)

            # Run classification if we have crops
            if crops_data:
                # jersey_classifier must be initialized in main()
                try:
                    classification_results = jersey_classifier.classify_batch(crops_data)
                except NameError:
                    print("Error: jersey_classifier is not initialized. Skipping classification for this batch.")
                    classification_results = [(None, 0.0, None, 0.0)] * len(crops_data)

                # Apply classification results
                crop_idx = 0
                for detection in frame_detections:
                    if detection['class_name'].lower() == 'player':
                        if crop_idx < len(classification_results):
                            color_class, color_conf, vis_class, vis_conf = classification_results[crop_idx]
                            detection['color_class'] = color_class
                            detection['color_conf'] = color_conf
                            detection['vis_class'] = vis_class
                            detection['vis_conf'] = vis_conf
                            crop_idx += 1

            # Store detections for this frame
            processed_frames.append({
                'frame': annotated_frame.copy(),
                'detections': frame_detections
            })

        else:
            # For non-YOLO frames, use previous detections
            if processed_frames:
                last_detections = processed_frames[-1]['detections']
                processed_frames.append({
                    'frame': annotated_frame.copy(),
                    'detections': last_detections
                })
            else:
                processed_frames.append({
                    'frame': annotated_frame.copy(),
                    'detections': []
                })

        # Progress indicator
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - {fps:.1f} FPS")

    cap.release()
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.1f} seconds")

    # Now render all frames with annotations and save
    print("Rendering annotated video...")
    for frame_idx, frame_data in enumerate(processed_frames):
        frame = frame_data['frame']
        detections = frame_data['detections']

        # Draw annotations
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if class_name.lower() == 'player':
                y_offset = 0
                if detection['color_class'] is not None:
                    color_label = f"Color: {detection['color_class']} ({detection['color_conf']:.2f})"
                    cv2.putText(frame, color_label, (x1, min(original_h - 5, y2 + 20 + y_offset)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += 20

                if detection['vis_class'] is not None:
                    vis_label = f"Number: {detection['vis_class']} ({detection['vis_conf']:.2f})"
                    cv2.putText(frame, vis_label, (x1, min(original_h - 5, y2 + 20 + y_offset)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            else:
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        writer.write(frame)

    writer.release()
    print(f"Annotated video saved to: {output_path}")

    # Play the processed video
    print("Playing processed video...")
    play_video(str(output_path))

    return True


def play_video(video_path):
    """Play the processed video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open processed video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay = 1.0 / fps

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Resize for display if needed
        if frame.shape[1] > 1750:
            display_w = 1750
            display_h = int(display_w * frame.shape[0] / frame.shape[1])
            frame = cv2.resize(frame, (display_w, display_h))

        cv2.imshow("Processed Video", frame)

        # Maintain proper playback speed
        elapsed = time.time() - start_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Determine project root (script lives in root/src/inference)
    script_dir = Path(__file__).resolve().parent  # root/src/inference
    root_dir = script_dir.parent.parent  # root
    config_dir = root_dir / "config"
    outputs_dir = config_dir / "outputs"
    config_path = config_dir / "inference.yaml"

    if not config_path.exists():
        print(f"Error: inference.yaml not found at expected location: {config_path}")
        print("Expected template (example):\n"
              "detection: a.pt\ncolor: b.pth\nvisibility: c.pth\nvideo path: path/to/video\n")
        sys.exit(1)

    # Load YAML
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if cfg is None:
                cfg = {}
    except Exception as e:
        print(f"Failed to read config '{config_path}': {e}")
        sys.exit(1)

    # Read keys (allow either 'video path' or 'video_path')
    detection_name = cfg.get('detection')
    color_name = cfg.get('color')
    visibility_name = cfg.get('visibility')
    video_path_cfg = cfg.get('video path') or cfg.get('video_path') or cfg.get('video')

    # Validate config keys
    if detection_name is None and color_name is None and visibility_name is None:
        print("Error: No model filenames found in inference.yaml (keys: detection, color, visibility).")
        sys.exit(1)

    if video_path_cfg is None:
        print("Error: No video path found in inference.yaml (key 'video path' or 'video_path').")
        sys.exit(1)

    # Build absolute paths to models (models are located in root/config/outputs)
    if outputs_dir.exists():
        print(f"Using outputs directory for models: {outputs_dir}")
    else:
        print(f"Outputs directory does not exist: {outputs_dir} (expected model files to be in this folder).")
        # still proceed, will check files individually

    # Detection model
    detection_path = None
    if detection_name:
        detection_path = outputs_dir / detection_name
        if not detection_path.exists():
            print(f"Detection model file not found at {detection_path}. Trying to use the name as absolute path...")
            maybe = Path(detection_name)
            if maybe.exists():
                detection_path = maybe
            else:
                print(f"Error: detection model '{detection_name}' not found in outputs or as absolute path.")
                detection_path = None

    # Color model
    color_path = None
    if color_name:
        color_path = outputs_dir / color_name
        if not color_path.exists():
            print(f"Color model file not found at {color_path}. Trying to use the name as absolute path...")
            maybe = Path(color_name)
            if maybe.exists():
                color_path = maybe
            else:
                print(f"Warning: color model '{color_name}' not found in outputs or as absolute path. Color classification disabled.")
                color_path = None

    # Visibility model
    visibility_path = None
    if visibility_name:
        visibility_path = outputs_dir / visibility_name
        if not visibility_path.exists():
            print(f"Visibility model file not found at {visibility_path}. Trying to use the name as absolute path...")
            maybe = Path(visibility_name)
            if maybe.exists():
                visibility_path = maybe
            else:
                print(f"Warning: visibility model '{visibility_name}' not found in outputs or as absolute path. Visibility classification disabled.")
                visibility_path = None

    # Resolve video path (allow relative to root_dir)
    video_path = Path(video_path_cfg)
    if not video_path.is_absolute():
        video_path = (root_dir / video_path_cfg).resolve()

    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    # Initialize detector (YOLO) if detection model is present
    if detection_path:
        try:
            print(f"Loading detection model from: {detection_path}")
            detector = YOLO(str(detection_path))
            # Warm-up: a small dummy inference to initialize model on device
            try:
                detector.predict(np.zeros((640, 640, 3), dtype=np.uint8))
            except Exception:
                # older ultralytics versions might want different warmup; ignore warmup failures
                pass
        except Exception as e:
            print(f"Failed to initialize YOLO detector from '{detection_path}': {e}")
            detector = None
    else:
        print("No detection model provided. Exiting.")
        sys.exit(1)

    # Initialize jersey classifier (color + visibility)
    jersey_classifier = JerseyClassifier(
        color_model_path=str(color_path) if color_path is not None else None,
        visibility_model_path=str(visibility_path) if visibility_path is not None else None
    )

    # Run processing
    print(f"Starting inference on video: {video_path}")
    success = process_video_offline(video_path)
    if not success:
        print("Processing failed or was aborted.")
    else:
        print("Processing finished successfully.")
