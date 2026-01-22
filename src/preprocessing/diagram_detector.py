import cv2
import numpy as np
import os
from PIL import Image

class DiagramDetector:
    """
    Detect and extract diagram/graph regions from handwritten notes
    """
    
    def __init__(self):
        self.output_dir = "temp/diagrams"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def detect_and_extract(self, image_path):
        """
        Detect diagrams in image and extract them
        
        Returns:
            dict with 'has_diagrams', 'diagram_regions', 'text_only_image'
        """
        print("🔍 Detecting diagrams...")
        
        img = cv2.imread(image_path)
        if img is None:
            return {'has_diagrams': False, 'diagram_regions': [], 'text_only_image': image_path}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Detect diagram regions
        diagram_regions = self._find_diagram_boxes(gray, img)
        
        if diagram_regions:
            print(f"  ✅ Found {len(diagram_regions)} diagram(s)")
            
            # Create text-only version (diagrams masked)
            text_only_path = self._create_text_only_image(img, diagram_regions)
            
            return {
                'has_diagrams': True,
                'diagram_regions': diagram_regions,
                'text_only_image': text_only_path,
                'original_image': image_path
            }
        else:
            print("  No diagrams detected")
            return {
                'has_diagrams': False,
                'diagram_regions': [],
                'text_only_image': image_path,
                'original_image': image_path
            }
    
    def _find_diagram_boxes(self, gray, img):
        """
        Find rectangular regions likely containing diagrams/graphs
        """
        # Preprocessing for contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = gray.shape
        min_area = (w * h) * 0.02  # At least 2% of image
        max_area = (w * h) * 0.5   # At most 50% of image
        
        diagram_regions = []
        
        for i, contour in enumerate(contours):
            x, y, w_box, h_box = cv2.boundingRect(contour)
            area = w_box * h_box
            
            # Filter by size and aspect ratio
            if min_area < area < max_area:
                aspect_ratio = w_box / h_box if h_box > 0 else 0
                
                # Diagrams usually have reasonable aspect ratios
                if 0.3 < aspect_ratio < 3.0:
                    roi = gray[y:y+h_box, x:x+w_box]
                    
                    # Check if region has graph-like features
                    if self._is_likely_diagram(roi):
                        # Extract diagram image
                        diagram_img = img[y:y+h_box, x:x+w_box]
                        diagram_path = os.path.join(self.output_dir, f"diagram_{i}.png")
                        cv2.imwrite(diagram_path, diagram_img)
                        
                        diagram_regions.append({
                            'id': i,
                            'bbox': (x, y, w_box, h_box),
                            'path': diagram_path,
                            'area': area,
                            'center_y': y + h_box // 2
                        })
        
        # Sort by vertical position
        diagram_regions.sort(key=lambda d: d['center_y'])
        
        return diagram_regions
    
    def _is_likely_diagram(self, roi):
        """
        Heuristic to determine if region contains a diagram/graph
        """
        # Check for straight lines (graphs have axes)
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=20,
            maxLineGap=5
        )
        
        if lines is None or len(lines) < 3:
            return False
        
        # Check for enclosed shapes (boxes, circles in diagrams)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Diagrams typically have multiple enclosed regions
        closed_shapes = sum(1 for cnt in contours if cv2.contourArea(cnt) > 100)
        
        # Heuristic: multiple lines + enclosed shapes = likely diagram
        return len(lines) >= 5 or closed_shapes >= 2
    
    def _create_text_only_image(self, img, diagram_regions):
        """
        Create version of image with diagrams masked out
        """
        text_only = img.copy()
        
        # Mask each diagram region with white
        # for region in diagram_regions:
        #     x, y, w, h = region['bbox']
        #     cv2.rectangle(text_only, (x, y), (x+w, y+h), (255, 255, 255), -1)
        
        text_only_path = "temp/text_only.png"
        cv2.imwrite(text_only_path, text_only)
        
        return text_only_path