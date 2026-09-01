import os
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import cv2
import numpy as np

class CowSegmenter:
    """
    DeepLabV3-ResNet50 Segmentation Engine
    """
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Initializing DeepLabV3 Cow Segmenter on device: {self.device}")
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights).to(self.device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cow_class_idx = 10  # Pascal VOC cow class

    def segment(self, image):
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            
        probs = torch.softmax(output, dim=0)
        cow_prob = probs[self.cow_class_idx].cpu().numpy()
        binary_mask = (cow_prob > 0.4).astype(np.uint8) * 255
        
        if binary_mask.shape != (h, w):
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        cleaned_mask = self._post_process(binary_mask)
        
        aspect_ratio = w / h if h > 0 else 0
        if np.sum(cleaned_mask == 255) < 0.05 * (h * w):
            if aspect_ratio >= 1.8 and h < 400:
                cleaned_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        return cleaned_mask

    def _post_process(self, mask):
        if np.sum(mask) == 0:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return closed
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=-1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        return clean_mask


class YOLOv8Segmenter:
    """
    YOLOv8 Segmentation Engine (Ultralytics)
    """
    def __init__(self, model_name='yolov8n-seg.pt'):
        print(f"Initializing YOLOv8 Segmenter with model: {model_name}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
        except Exception as e:
            print(f"Error loading YOLOv8: {e}")
            self.model = None

    def segment(self, image):
        if self.model is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
        h, w, _ = image.shape
        results = self.model(image, verbose=False)[0]
        
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        
        if results.masks is not None:
            # COCO class 19 is 'cow'
            cow_masks = []
            for i, cls_idx in enumerate(results.boxes.cls):
                if int(cls_idx) == 19:  # Cow class in COCO
                    mask_data = results.masks.data[i].cpu().numpy()
                    cow_masks.append(mask_data)
                    
            if len(cow_masks) > 0:
                # Combine all cow masks or take the largest one
                largest_mask = max(cow_masks, key=lambda m: np.sum(m))
                binary_mask = (largest_mask > 0.5).astype(np.uint8) * 255
            else:
                # If no specific cow class detected, take the largest object mask in the image
                all_masks = [m.cpu().numpy() for m in results.masks.data]
                if len(all_masks) > 0:
                    largest_mask = max(all_masks, key=lambda m: np.sum(m))
                    binary_mask = (largest_mask > 0.5).astype(np.uint8) * 255
                    
        if binary_mask.shape != (h, w):
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        return binary_mask


if __name__ == "__main__":
    segmenter = CowSegmenter()
    print("CowSegmenter (DeepLabV3) ready.")
    yolo_segmenter = YOLOv8Segmenter()
    print("YOLOv8Segmenter ready.")
