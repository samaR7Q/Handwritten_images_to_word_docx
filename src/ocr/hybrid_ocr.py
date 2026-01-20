import os
from dotenv import load_dotenv

load_dotenv()

class HybridOCR:
    """
    Intelligent OCR that tries API first, falls back to local model
    
    Priority:
    1. Groq Llama Vision API (fast, good quality)
    2. Florence-2 Local (offline, reliable)
    3. EasyOCR (last resort fallback)
    """
    
    def __init__(self, prefer_local=False):
        """
        Args:
            prefer_local: If True, use local model first (skip API)
        """
        self.prefer_local = prefer_local
        
        print(f"🔧 Initializing Hybrid OCR (prefer_local={prefer_local})...")
        
        # Initialize API OCR
        self.api_ocr = None
        if not prefer_local:
            try:
                from .vision_ocr import VisionOCR
                groq_key = os.getenv('GROQ_API_KEY')
                if groq_key:
                    self.api_ocr = VisionOCR()
                    print("✅ API OCR available (Groq Llama Vision)")
                else:
                    print("⚠️  No Groq API key found")
            except Exception as e:
                print(f"⚠️  API OCR unavailable: {e}")
        
        # Initialize local Florence-2
        self.local_ocr = None
        try:
            from .florence_local_ocr import FlorenceLocalOCR
            self.local_ocr = FlorenceLocalOCR()
        except Exception as e:
            print(f"⚠️  Local OCR unavailable: {e}")
        
        # Initialize EasyOCR as last fallback
        self.easy_ocr = None
        try:
            import easyocr
            self.easy_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            self.easy_ocr = True
            print("✅ EasyOCR available (fallback)")
        except Exception as e:
            print(f"⚠️  EasyOCR unavailable: {e}")
        
        print("✅ Hybrid OCR initialized")
    
    def extract_text_from_image(self, image_path):
        """
        Extract text with intelligent fallback
        
        Returns:
            dict with 'text', 'confidence', 'method'
        """
        result = None
        
        # Strategy 1: Try API if available and not preferring local
        if not self.prefer_local and self.api_ocr:
            print("  📡 Trying API OCR (Groq Llama Vision)...")
            try:
                result = self.api_ocr.extract_text_from_image(image_path)
                
                # Check if API succeeded
                if result.get('text') and len(result['text']) > 50:
                    print(f"  ✅ API OCR succeeded ({result.get('method', 'api')})")
                    return result
                else:
                    print("  ⚠️  API returned minimal text, trying local...")
                    
            except Exception as e:
                print(f"  ❌ API OCR failed: {e}")
                print("  🔄 Falling back to local model...")
        
        # Strategy 2: Try local Florence-2
        if self.local_ocr and self.local_ocr.available:
            print("  🖥️  Using Local Florence-2 OCR...")
            try:
                result = self.local_ocr.extract_text_from_image(image_path)
                
                if result.get('text') and len(result['text']) > 50:
                    print(f"  ✅ Local OCR succeeded (Florence-2)")
                    return result
                else:
                    print("  ⚠️  Florence-2 returned minimal text, trying EasyOCR...")
                    
            except Exception as e:
                print(f"  ❌ Local OCR failed: {e}")
        
        # Strategy 3: Last resort - EasyOCR
        if self.easy_ocr:
            print("  🔄 Using EasyOCR (last resort)...")
            try:
                result = self._easy_ocr_extract(image_path)
                if result.get('text'):
                    print(f"  ✅ EasyOCR succeeded")
                    return result
            except Exception as e:
                print(f"  ❌ EasyOCR failed: {e}")
        
        # All methods failed
        print("  ❌ All OCR methods failed!")
        return {
            "text": "",
            "confidence": 0,
            "error": "All OCR methods failed",
            "method": "all_failed"
        }
    
    def _easy_ocr_extract(self, image_path):
        """Fallback EasyOCR extraction"""
        try:
            result = self.easy_reader.readtext(image_path, detail=1, paragraph=False)
            
            if not result:
                return {"text": "", "confidence": 0}
            
            # Sort and combine
            result_sorted = sorted(result, key=lambda x: (x[0][0][1], x[0][0][0]))
            
            lines = []
            current_y = -1
            line_threshold = 30
            
            for (bbox, text, conf) in result_sorted:
                y_pos = bbox[0][1]
                
                if current_y == -1 or abs(y_pos - current_y) > line_threshold:
                    if lines:
                        lines.append('\n')
                    current_y = y_pos
                else:
                    lines.append(' ')
                
                lines.append(text)
            
            full_text = ''.join(lines)
            avg_confidence = sum([conf for (_, _, conf) in result]) / len(result)
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "method": "easyocr_fallback"
            }
            
        except Exception as e:
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def cleanup(self):
        """Clean up resources"""
        if self.local_ocr:
            self.local_ocr.cleanup()