import cv2
import numpy as np
from PIL import Image
import os

class ImagePreprocessor:
    """Handles image preprocessing for OCR optimization"""
    
    def __init__(self):
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def preprocess(self, image_path):
        """
        Main preprocessing pipeline
        Returns: preprocessed image (numpy array)
        """
        print(f"📸 Preprocessing image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize if too large (keep aspect ratio)
        height, width = img.shape[:2]
        max_dimension = 2000
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            print(f"  Resized to: {new_width}x{new_height}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise (lighter touch)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
        
        # Adaptive thresholding (better for handwriting than Otsu)
        binary = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Invert if background is dark
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # Save preprocessed image
        output_path = os.path.join(self.temp_dir, "preprocessed.png")
        cv2.imwrite(output_path, binary)
        
        print(f"✅ Preprocessing complete: {output_path}")
        return binary, output_path
    
    def extract_regions(self, image):
        """
        SIMPLIFIED: Just return the whole image as one text region
        Better for dense handwritten notes
        """
        print("  Using full-page OCR (better for dense handwriting)")
        
        regions = [{
            'id': 0,
            'bbox': (0, 0, image.shape[1], image.shape[0]),
            'type': 'full_page',
            'image': image
        }]
        
        return regions