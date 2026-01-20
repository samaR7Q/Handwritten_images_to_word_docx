import os
import requests
from dotenv import load_dotenv
import base64
import cv2
import numpy as np

load_dotenv()

class OCREngine:
    """Handles OCR using multiple engines"""
    
    def __init__(self):
        self.groq_key = os.getenv('GROQ_API_KEY')
        self.mathpix_id = os.getenv('MATHPIX_APP_ID')
        self.mathpix_key = os.getenv('MATHPIX_APP_KEY')
        
        # Initialize EasyOCR with better settings
        try:
            import easyocr
            self.easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("✅ EasyOCR initialized")
        except Exception as e:
            self.easy_reader = None
            print(f"⚠️ EasyOCR error: {e}")
    
    def extract_text(self, image_path, region_type='text_block'):
        """
        Extract text from image
        """
        # For full page, use comprehensive extraction
        if region_type == 'full_page':
            return self._extract_full_page(image_path)
        elif region_type == 'diagram':
            return self._extract_diagram(image_path)
        else:
            return self._extract_handwriting(image_path)
    
    def _extract_full_page(self, image_path):
        """Extract all text from full page image"""
        print(f"  Running full-page OCR...")
        
        if self.easy_reader is None:
            return {"text": "[OCR Error: EasyOCR not initialized]", "confidence": 0}
        
        try:
            # Run EasyOCR with paragraph mode
            result = self.easy_reader.readtext(
                image_path, 
                detail=1,
                paragraph=False,  # Get individual text blocks
                width_ths=0.7,    # Adjust text block width threshold
                height_ths=0.7    # Adjust text block height threshold
            )
            
            if not result:
                print("  ⚠️ No text detected!")
                return {"text": "", "confidence": 0, "details": []}
            
            # Sort by vertical position (top to bottom, left to right)
            result_sorted = sorted(result, key=lambda x: (x[0][0][1], x[0][0][0]))
            
            # Combine text with line breaks
            lines = []
            current_y = -1
            line_threshold = 30  # pixels
            
            for (bbox, text, conf) in result_sorted:
                y_pos = bbox[0][1]
                
                # Check if new line
                if current_y == -1 or abs(y_pos - current_y) > line_threshold:
                    if lines and lines[-1]:  # Add newline if not empty
                        lines.append('\n')
                    current_y = y_pos
                else:
                    lines.append(' ')  # Same line, add space
                
                lines.append(text)
            
            full_text = ''.join(lines)
            avg_confidence = sum([conf for (bbox, text, conf) in result]) / len(result)
            
            print(f"  ✅ Extracted {len(result)} text blocks (avg confidence: {avg_confidence:.2f})")
            print(f"  Preview: {full_text[:100]}...")
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "details": result
            }
            
        except Exception as e:
            print(f"  ❌ EasyOCR error: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def _extract_handwriting(self, image_path):
        """Extract handwritten text using EasyOCR"""
        if self.easy_reader is None:
            return {"text": "[OCR Error: EasyOCR not initialized]", "confidence": 0}
        
        try:
            result = self.easy_reader.readtext(image_path, detail=1)
            
            if not result:
                return {"text": "", "confidence": 0, "details": []}
            
            # Combine all text
            full_text = ' '.join([text for (bbox, text, conf) in result])
            avg_confidence = sum([conf for (bbox, text, conf) in result]) / len(result)
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "details": result
            }
        except Exception as e:
            print(f"❌ EasyOCR error: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def _extract_math(self, image_path):
        """Extract mathematical equations using Mathpix"""
        if not self.mathpix_id or not self.mathpix_key:
            print("⚠️ Mathpix API not configured, using EasyOCR")
            return self._extract_handwriting(image_path)
        
        try:
            with open(image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            
            headers = {
                'app_id': self.mathpix_id,
                'app_key': self.mathpix_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'src': f'data:image/png;base64,{img_base64}',
                'formats': ['text', 'latex_simplified'],
                'ocr': ['math', 'text']
            }
            
            response = requests.post(
                'https://api.mathpix.com/v3/text',
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "text": result.get('text', ''),
                    "latex": result.get('latex_simplified', ''),
                    "confidence": result.get('confidence', 0)
                }
            else:
                print(f"⚠️ Mathpix API error: {response.status_code}")
                return self._extract_handwriting(image_path)
                
        except Exception as e:
            print(f"❌ Mathpix error: {e}")
            return self._extract_handwriting(image_path)
    
    def _extract_diagram(self, image_path):
        """For diagrams, we'll just mark them for embedding"""
        return {
            "text": "[DIAGRAM]",
            "image_path": image_path,
            "type": "diagram"
        }
    
    def _has_math_symbols(self, image_path):
        """Simple heuristic to detect if image might contain math"""
        if self.easy_reader:
            result = self.easy_reader.readtext(image_path)
            text = ' '.join([t for (bbox, t, conf) in result])
            math_indicators = ['=', '+', '-', '×', '÷', 'Δ', '→', '∫', '∂', 'Σ', '∑']
            return any(symbol in text for symbol in math_indicators)
        return False