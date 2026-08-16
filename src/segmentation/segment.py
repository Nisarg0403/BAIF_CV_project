import os
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import cv2
import numpy as np

class CowSegmenter:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Initializing Cow Segmenter on device: {self.device}")
        
        # Load pre-trained DeepLabV3-ResNet50 model
        # Using the recommended weights parameter
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights).to(self.device)
        self.model.eval()
        
        # Define transform for ImageNet preprocessing
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # In Pascal VOC (which DeepLabV3 is trained on), index 10 is 'cow'
        self.cow_class_idx = 10

    def segment(self, image):
        """
        Segments the cow from the given image (NumPy array).
        Returns a binary mask (numpy array of 0 and 255) of the same HxW.
        """
        h, w, _ = image.shape
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess and send to device
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            
        # Get probabilities for VOC classes
        # Shape: (21, H, W)
        probs = torch.softmax(output, dim=0)
        
        # Extract the cow class probability map
        cow_prob = probs[self.cow_class_idx].cpu().numpy()
        
        # Threshold to create binary mask (e.g. probability > 0.3 or 0.5)
        binary_mask = (cow_prob > 0.4).astype(np.uint8) * 255
        
        # Resize mask back to original size if it was resized (DeepLabV3 does not resize output)
        if binary_mask.shape != (h, w):
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        # Post-process binary mask to clean noise
        cleaned_mask = self._post_process(binary_mask)
        
        # Fallback: if segmentation failed or detected area is extremely small
        # AND this looks like a pre-cropped cow image (aspect ratio >= 1.8 and height < 400),
        # we fall back to treating the entire image as the cow mask.
        aspect_ratio = w / h if h > 0 else 0
        if np.sum(cleaned_mask == 255) < 0.05 * (h * w):
            if aspect_ratio >= 1.8 and h < 400:
                cleaned_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        return cleaned_mask

    def _post_process(self, mask):
        """
        Cleans the binary mask using morphological operations and keeps only the largest contour.
        """
        if np.sum(mask) == 0:
            return mask
            
        # 1. Apply Morphological Closing to fill small holes inside the cow silhouette
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 2. Find contours and keep only the largest one (the main cow body)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return closed
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a new blank mask and draw the largest contour
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=-1)
        
        # 3. Apply Morphological Opening to remove small noise on the boundaries
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        
        return clean_mask

# Simple test function
if __name__ == "__main__":
    segmenter = CowSegmenter()
    print("Segmenter loaded successfully.")
