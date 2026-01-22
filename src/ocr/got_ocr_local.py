import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os

class GOTOCRLocal:
    """
    Local OCR using GOT-OCR 2.0 (General OCR Theory)
    State-of-the-art model for document OCR with formulas and handwriting
    """
    
    def __init__(self, device=None):
        """
        Initialize GOT-OCR 2.0 model
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        print("🔧 Initializing GOT-OCR 2.0...")
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"   Device: {self.device}")
        
        try:
            # GOT-OCR 2.0 model
            model_name = "stepfun-ai/GOT-OCR2_0"
            
            print(f"   Loading {model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=self.device if self.device == "cuda" else None,
                use_safetensors=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Move to device if CPU
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set to eval mode
            self.model.eval()
            
            print("✅ GOT-OCR 2.0 loaded successfully")
            self.available = True
            
        except Exception as e:
            print(f"❌ Failed to load GOT-OCR 2.0: {e}")
            print(f"   Error details: {str(e)}")
            print("   Will use fallback methods if called")
            self.available = False
    
    def extract_text_from_image(self, image_path, ocr_type='ocr'):
        """
        Extract text from image using GOT-OCR 2.0
        
        Args:
            image_path: Path to image file
            ocr_type: 'ocr' (plain text), 'format' (with formatting), or 'formula' (LaTeX formulas)
            
        Returns:
            dict with 'text', 'confidence', 'method'
        """
        if not self.available:
            return {
                "text": "",
                "confidence": 0,
                "error": "GOT-OCR 2.0 model not available",
                "method": "got_ocr_failed"
            }
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            print(f"  🖼️  Processing with GOT-OCR 2.0 ({ocr_type} mode)...")
            
            # GOT-OCR 2.0 supports different OCR modes
            # 'ocr' - plain OCR
            # 'format' - OCR with format preservation
            # 'formula' - Focus on mathematical formulas
            
            # For handwritten chemistry notes, use 'format' mode
            with torch.no_grad():
                # GOT-OCR uses chat method for inference
                result = self.model.chat(
                    self.tokenizer,
                    image_path,
                    ocr_type=ocr_type  # 'ocr', 'format', or 'formula'
                )
            
            # GOT-OCR returns plain text
            extracted_text = result.strip() if isinstance(result, str) else str(result)
            
            print(f"  ✅ GOT-OCR extracted {len(extracted_text)} characters")
            print(f"     Preview: {extracted_text[:100]}...")
            
            # GOT-OCR doesn't provide confidence scores
            # Estimate based on output length and structure
            confidence = self._estimate_confidence(extracted_text)
            
            return {
                "text": extracted_text,
                "confidence": confidence,
                "method": "got_ocr_local",
                "ocr_type": ocr_type
            }
            
        except Exception as e:
            print(f"  ❌ GOT-OCR error: {e}")
            return {
                "text": "",
                "confidence": 0,
                "error": str(e),
                "method": "got_ocr_error"
            }
    
    def extract_with_box_detection(self, image_path):
        """
        Extract text with bounding box detection
        GOT-OCR 2.0 can detect text regions and extract them
        """
        if not self.available:
            return {"text": "", "confidence": 0}
        
        try:
            print(f"  🔍 Processing with box detection...")
            
            with torch.no_grad():
                # Use 'format' mode for better structure preservation
                result = self.model.chat(
                    self.tokenizer,
                    image_path,
                    ocr_type='format',
                    ocr_box=''  # Empty string enables box detection
                )
            
            extracted_text = result.strip() if isinstance(result, str) else str(result)
            confidence = self._estimate_confidence(extracted_text)
            
            return {
                "text": extracted_text,
                "confidence": confidence,
                "method": "got_ocr_boxes"
            }
            
        except Exception as e:
            print(f"  ❌ Box detection error: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def extract_formulas(self, image_path):
        """
        Extract mathematical formulas in LaTeX format
        Useful for chemistry equations
        """
        if not self.available:
            return {"text": "", "confidence": 0}
        
        try:
            print(f"  🧮 Extracting formulas...")
            
            with torch.no_grad():
                result = self.model.chat(
                    self.tokenizer,
                    image_path,
                    ocr_type='formula'  # Formula mode for LaTeX
                )
            
            extracted_text = result.strip() if isinstance(result, str) else str(result)
            confidence = self._estimate_confidence(extracted_text)
            
            return {
                "text": extracted_text,
                "confidence": confidence,
                "method": "got_ocr_formula",
                "format": "latex"
            }
            
        except Exception as e:
            print(f"  ❌ Formula extraction error: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def _estimate_confidence(self, text):
        """
        Estimate confidence based on output characteristics
        GOT-OCR doesn't provide confidence scores, so we estimate
        """
        if not text:
            return 0.0
        
        # Base confidence
        confidence = 0.85
        
        # Adjust based on text length (very short = suspicious)
        if len(text) < 10:
            confidence -= 0.3
        elif len(text) < 50:
            confidence -= 0.1
        
        # Adjust based on special characters (formulas, etc.)
        special_chars = sum(1 for c in text if c in '∆→≈≠±×÷∫∑')
        if special_chars > 5:
            confidence += 0.05  # Likely chemistry/math content
        
        # Clip to valid range
        return max(0.0, min(1.0, confidence))
    
    def cleanup(self):
        """Free up memory"""
        if self.available:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ GOT-OCR 2.0 model unloaded")