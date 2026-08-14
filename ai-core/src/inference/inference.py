import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Cố định phần cứng là CPU cho môi trường Azure ML Serving
device = torch.device('cpu')

# Giữ nguyên cấu hình tiền xử lý chuẩn cho ResNet từ mã gốc
preprocess = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class JerseyClassifier:
    """Trình phân loại màu sắc và độ hiển thị số áo"""

    def __init__(self, color_model_path, visibility_model_path):
        # Thiết lập thiết bị tính toán cục bộ, loại bỏ phụ thuộc global variable
        self.device = device 
        self.color_model = None
        self.color_classes = []
        self.visibility_model = None
        self.visibility_classes = []

        if color_model_path and Path(color_model_path).exists():
            color_checkpoint = torch.load(color_model_path, map_location=self.device)
            self.color_classes = color_checkpoint['class_names']
            self.color_model = models.resnet50(pretrained=False)
            self.color_model.fc = nn.Linear(self.color_model.fc.in_features, len(self.color_classes))
            self.color_model.load_state_dict(color_checkpoint['model_state_dict'])
            self.color_model = self.color_model.to(self.device)
            self.color_model.eval()

        if visibility_model_path and Path(visibility_model_path).exists():
            vis_checkpoint = torch.load(visibility_model_path, map_location=self.device)
            self.visibility_classes = vis_checkpoint['class_names']
            self.visibility_model = models.resnet50(pretrained=False)
            self.visibility_model.fc = nn.Linear(self.visibility_model.fc.in_features, len(self.visibility_classes))
            self.visibility_model.load_state_dict(vis_checkpoint['model_state_dict'])
            self.visibility_model = self.visibility_model.to(self.device)
            self.visibility_model.eval()

    def classify_batch(self, crop_images):
        """Phân loại một lô ảnh cắt (crop)"""
        if not crop_images:
            return []

        batch_tensors = []
        valid_indices = []

        for i, crop_image in enumerate(crop_images):
            if crop_image is None or crop_image.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(Image.fromarray(crop_rgb))
            batch_tensors.append(input_tensor)
            valid_indices.append(i)

        if not batch_tensors:
            return [(None, 0.0, None, 0.0)] * len(crop_images)

        input_batch = torch.stack(batch_tensors).to(self.device)
        results = [(None, 0.0, None, 0.0)] * len(crop_images)

        with torch.no_grad():
            if self.color_model:
                c_out = self.color_model(input_batch)
                c_top_prob, c_top_idx = torch.topk(torch.nn.functional.softmax(c_out, dim=1), 1, dim=1)
            
            if self.visibility_model:
                v_out = self.visibility_model(input_batch)
                v_top_prob, v_top_idx = torch.topk(torch.nn.functional.softmax(v_out, dim=1), 1, dim=1)

            for i, valid_idx in enumerate(valid_indices):
                c_class = self.color_classes[c_top_idx[i].item()] if self.color_model else None
                c_conf = c_top_prob[i].item() if self.color_model else 0.0
                v_class = self.visibility_classes[v_top_idx[i].item()] if self.visibility_model else None
                v_conf = v_top_prob[i].item() if self.visibility_model else 0.0
                results[valid_idx] = (c_class, c_conf, v_class, v_conf)

        return results

def box_iou(boxA, boxB):
    """Tính toán IoU giữa hai bounding box"""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0