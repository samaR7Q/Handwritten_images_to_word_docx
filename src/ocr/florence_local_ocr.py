import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import os

class FlorenceLocalOCR:
    """
    Local OCR using Microsoft's Florence-2-base model
    Runs on 6GB VRAM, excellent for handwritten documents
    """
    
    def __init__(self, device=None):
        """
        Initialize Florence-2 model
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        print("🔧 Initializing Florence-2 Local OCR...")
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"   Device: {self.device}")
        
        try:
            # Load model and processor
            model_name = "microsoft/Florence-2-base"
            
            print(f"   Loading {model_name}...")
            self.processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            print("✅ Florence-2 loaded successfully")
            self.available = True
            
        except Exception as e:
            print(f"❌ Failed to load Florence-2: {e}")
            print("   Will use fallback methods if called")
            self.available = False
    
    def extract_text_from_image(self, image_path):
        """
        Extract text from image using Florence-2
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with 'text', 'confidence', 'method'
        """
        if not self.available:
            return {
                "text": "",
                "confidence": 0,
                "error": "Florence-2 model not available",
                "method": "florence_local_failed"
            }
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            print(f"  🖼️  Processing with Florence-2...")
            
            # Task: OCR with region understanding
            # Florence-2 supports multiple tasks
            task_prompt = "<OCR_WITH_REGION>"
            
            # Process image
            inputs = self.processor(
                text=task_prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move to device and convert dtype appropriately
            # input_ids must stay as Long, pixel_values should match model dtype
            inputs = {
                k: v.to(self.device).to(self.model.dtype) if k == "pixel_values" and isinstance(v, torch.Tensor)
                else v.to(self.device) if isinstance(v, torch.Tensor)
                else v
                for k, v in inputs.items()
            }
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=2048,
                    num_beams=3,
                    do_sample=False
                )
            
            # Decode
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Post-process Florence output
            parsed_text = self._parse_florence_output(generated_text, task_prompt)
            
            print(f"  ✅ Florence-2 extracted {len(parsed_text)} characters")
            print(f"     Preview: {parsed_text[:100]}...")
            
            return {
                "text": parsed_text,
                "confidence": 0.85,  # Florence is quite reliable
                "method": "florence_local",
                "raw_output": generated_text
            }
            
        except Exception as e:
            print(f"  ❌ Florence-2 error: {e}")
            return {
                "text": "",
                "confidence": 0,
                "error": str(e),
                "method": "florence_local_error"
            }
    
    def extract_with_detailed_caption(self, image_path):
        """
        Alternative: Get detailed description (useful for diagrams)
        """
        if not self.available:
            return {"text": "", "confidence": 0}
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Use detailed caption task
            task_prompt = "<MORE_DETAILED_CAPTION>"
            
            inputs = self.processor(
                text=task_prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move to device and convert dtype appropriately
            # input_ids must stay as Long, pixel_values should match model dtype
            inputs = {
                k: v.to(self.device).to(self.model.dtype) if k == "pixel_values" and isinstance(v, torch.Tensor)
                else v.to(self.device) if isinstance(v, torch.Tensor)
                else v
                for k, v in inputs.items()
            }
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
            
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            parsed = self._parse_florence_output(generated_text, task_prompt)
            
            return {
                "text": parsed,
                "confidence": 0.80,
                "method": "florence_caption"
            }
            
        except Exception as e:
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def _parse_florence_output(self, output, task_prompt):
        """
        Parse Florence-2 output format
        Florence returns structured data that needs cleaning
        """
        # Remove task prompt from output
        cleaned = output.replace(task_prompt, "").strip()
        
        # Florence with OCR_WITH_REGION returns JSON-like structure
        # Extract just the text content
        try:
            # If it's structured output, extract text regions
            if "quad_boxes" in cleaned or "labels" in cleaned:
                # Parse the labels (actual text)
                import re
                # Extract text between quotes in labels
                texts = re.findall(r"'([^']*)'", cleaned)
                if texts:
                    return " ".join(texts)
            
            # Otherwise return as-is
            return cleaned
            
        except:
            return cleaned
    
    def cleanup(self):
        """Free up memory"""
        if self.available:
            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ Florence-2 model unloaded")