import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class VisionOCR:
    """Use multimodal LLM for better handwriting OCR"""
    
    def __init__(self):
        self.groq_key = os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=self.groq_key) if self.groq_key else None
        print("✅ Vision OCR initialized")
    
    def extract_text_from_image(self, image_path):
        """
        Use Llama 3.2 Vision (via Groq) to extract text
        """
        if not self.client:
            print("❌ Groq API not available")
            return {"text": "", "confidence": 0}
        
        try:
            # Read and encode image
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Groq supports Llama 3.2 Vision models
            completion = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",  # Vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Extract ALL text from this handwritten chemistry notes image.

Rules:
1. Transcribe EXACTLY what you see, including all text, equations, and formulas
2. Preserve structure: headings, bullet points, boxes
3. Use markdown formatting (# for headings, * for bullets)
4. For chemical formulas, use subscripts as numbers (H2O not H₂O)
5. Preserve all mathematical symbols and equations
6. For diagrams/graphs:
   - Mark their location with [DIAGRAM: brief description]
   - Example: [DIAGRAM: Graph showing pressure vs α/m with curves for different n values]
   - Describe axes, labels, and key features
7. Maintain the reading order (top to bottom, left to right)
8. Don't skip anything - transcribe everything visible
9: Do not return answers in LaTeX format.
    Write all mathematical formulas in plain text, using standard symbols.
    For example:
        Fractions as a / b
        Powers as x^2
        Subscripts as I_DC
        Integrals as integral(expression, variable, lower, upper) if needed
        Piecewise functions using words or simple if-then notation
Output only the transcribed text with diagram descriptions."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            extracted_text = completion.choices[0].message.content.strip()
            
            print(f"✅ Vision OCR complete")
            print(f"   Extracted {len(extracted_text)} characters")
            print(f"   Preview: {extracted_text[:100]}...")
            
            return {
                "text": extracted_text,
                "confidence": 0.95,  # Vision models are much better
                "method": "llama_vision"
            }
            
        except Exception as e:
            print(f"❌ Vision OCR error: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}