import os
from dotenv import load_dotenv

load_dotenv()

class HybridOCR:
    """
    Intelligent OCR with 4 options:
    1. Llama Vision API (Groq)
    2. Florence-2 Local
    3. GOT-OCR 2.0 Local (Best for handwriting + formulas)
    4. EasyOCR Fallback
    """
    
    def __init__(self, prefer_local=False, local_model="auto"):
        """
        Args:
            prefer_local: If True, skip API and use local model
            local_model: "auto", "florence", "got", or "easyocr"
        """
        self.prefer_local = prefer_local
        self.local_model = local_model
        
        print(f"🔧 Initializing Hybrid OCR...")
        print(f"   Mode: {'LOCAL' if prefer_local else 'API-FIRST'}")
        print(f"   Local Model: {local_model}")
        
        # Initialize API OCR
        self.api_ocr = None
        if not prefer_local:
            try:
                from .vision_ocr import VisionOCR
                groq_key = os.getenv('GROQ_API_KEY')
                if groq_key:
                    self.api_ocr = VisionOCR()
                    print("✅ API OCR available (Llama Vision)")
                else:
                    print("⚠️  No Groq API key found")
            except Exception as e:
                print(f"⚠️  API OCR unavailable: {e}")
        
        # Lazy loading for local models (only load when needed)
        self.florence_ocr = None
        self.got_ocr = None
        self.easy_ocr = None
        self.easy_reader = None
        
        print("✅ Hybrid OCR initialized (local models will load on demand)")
    
    def extract_text_from_image(self, image_path):
        """
        Extract text with intelligent fallback based on initialization settings
        
        Returns:
            dict with 'text', 'confidence', 'method'
        """
        result = None
        
        # Force specific model if requested during init
        if self.local_model == "api":
            if not self.api_ocr:
                print("  ⚠️  API OCR not available, falling back...")
            else:
                return self._try_api_ocr(image_path)
        
        elif self.local_model == "florence":
            result = self._try_florence(image_path)
            if self._is_good_result(result):
                return result
            print("  ⚠️  Florence failed, trying fallback...")
        
        elif self.local_model == "got":
            result = self._try_got_ocr(image_path)
            if self._is_good_result(result):
                return result
            print("  ⚠️  GOT-OCR failed, trying fallback...")
        
        elif self.local_model == "easyocr":
            return self._try_easyocr(image_path)
        
        # Auto mode: try in priority order
        if not self.prefer_local and self.api_ocr:
            result = self._try_api_ocr(image_path)
            if self._is_good_result(result):
                return result
        
        # Try Florence
        result = self._try_florence(image_path)
        if self._is_good_result(result):
            return result
        
        # Try GOT-OCR
        result = self._try_got_ocr(image_path)
        if self._is_good_result(result):
            return result
        
        # Last resort: EasyOCR
        result = self._try_easyocr(image_path)
        if self._is_good_result(result):
            return result
        
        # All failed
        print("  ❌ All OCR methods failed!")
        return {
            "text": "",
            "confidence": 0,
            "error": "All OCR methods failed",
            "method": "all_failed"
        }
    
    def _try_api_ocr(self, image_path):
        """Try Llama Vision API"""
        print("  📡 Trying Llama Vision API...")
        try:
            result = self.api_ocr.extract_text_from_image(image_path)
            if result.get('text'):
                print(f"  ✅ API OCR succeeded")
            return result
        except Exception as e:
            print(f"  ❌ API OCR error: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def _try_florence(self, image_path):
        """Try Florence-2 (lazy load)"""
        if not self.florence_ocr:
            print("  📥 Loading Florence-2 model...")
            try:
                from .florence_local_ocr import FlorenceLocalOCR
                self.florence_ocr = FlorenceLocalOCR()
            except Exception as e:
                print(f"  ❌ Failed to load Florence-2: {e}")
                return {"text": "", "confidence": 0, "error": str(e)}
        
        print("  🖥️  Trying Florence-2...")
        try:
            result = self.florence_ocr.extract_text_from_image(image_path)
            if result.get('text'):
                print(f"  ✅ Florence-2 succeeded")
            return result
        except Exception as e:
            print(f"  ❌ Florence-2 error: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def _try_got_ocr(self, image_path):
        """Try GOT-OCR 2.0 (lazy load)"""
        if not self.got_ocr:
            print("  📥 Loading GOT-OCR 2.0 model...")
            try:
                from .got_ocr_local import GOTOCRLocal
                self.got_ocr = GOTOCRLocal()
            except Exception as e:
                print(f"  ❌ Failed to load GOT-OCR 2.0: {e}")
                return {"text": "", "confidence": 0, "error": str(e)}
        
        print("  🖥️  Trying GOT-OCR 2.0...")
        try:
            result = self.got_ocr.extract_text_from_image(image_path, ocr_type='format')
            if result.get('text'):
                print(f"  ✅ GOT-OCR succeeded")
            return result
        except Exception as e:
            print(f"  ❌ GOT-OCR error: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def _try_easyocr(self, image_path):
        """Try EasyOCR (lazy load)"""
        if not self.easy_reader:
            print("  📥 Loading EasyOCR...")
            try:
                import easyocr
                self.easy_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                self.easy_ocr = True
            except Exception as e:
                print(f"  ❌ Failed to load EasyOCR: {e}")
                return {"text": "", "confidence": 0, "error": str(e)}
        
        print("  🔄 Trying EasyOCR...")
        try:
            result = self.easy_reader.readtext(image_path, detail=1, paragraph=False)
            
            if not result:
                return {"text": "", "confidence": 0}
            
            result_sorted = sorted(result, key=lambda x: (x[0][0][1], x[0][0][0]))
            
            lines = []
            current_y = -1
            
            for (bbox, text, conf) in result_sorted:
                y_pos = bbox[0][1]
                
                if current_y == -1 or abs(y_pos - current_y) > 30:
                    if lines:
                        lines.append('\n')
                    current_y = y_pos
                else:
                    lines.append(' ')
                
                lines.append(text)
            
            full_text = ''.join(lines)
            avg_confidence = sum([conf for (_, _, conf) in result]) / len(result)
            
            print(f"  ✅ EasyOCR succeeded")
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "method": "easyocr"
            }
            
        except Exception as e:
            print(f"  ❌ EasyOCR error: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def _is_good_result(self, result):
        """Check if OCR result is acceptable"""
        return (result.get('text') and 
                len(result['text']) > 50 and 
                result.get('confidence', 0) > 0.1)
    
    def cleanup(self):
        """Clean up all models that were actually loaded"""
        cleaned = []
        
        if self.florence_ocr and hasattr(self.florence_ocr, 'available') and self.florence_ocr.available:
            try:
                self.florence_ocr.cleanup()
                cleaned.append("Florence-2")
            except:
                pass
        
        if self.got_ocr and hasattr(self.got_ocr, 'available') and self.got_ocr.available:
            try:
                self.got_ocr.cleanup()
                cleaned.append("GOT-OCR 2.0")
            except:
                pass
        
        if self.easy_ocr and hasattr(self, 'easy_reader'):
            try:
                del self.easy_reader
                cleaned.append("EasyOCR")
            except:
                pass
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if cleaned:
            print(f"✅ Cleaned up: {', '.join(cleaned)}")
        else:
            print("✅ Cleanup complete (no models were loaded)")