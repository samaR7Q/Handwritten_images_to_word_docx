import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMCorrector:
    """Uses Groq + Llama for intelligent OCR correction"""
    
    def __init__(self):
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("⚠️ Groq API key not found")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
            print("✅ Groq client initialized")
    
    def correct_text(self, ocr_text, context="chemistry notes"):
        """
        Correct OCR errors using LLM
        """
        # Don't process empty text
        if not ocr_text or len(ocr_text.strip()) < 10:
            print("⚠️ OCR text too short or empty, skipping LLM correction")
            return ocr_text
        
        if not self.client:
            print("⚠️ Groq not available, returning uncorrected text")
            return ocr_text
        
        prompt = f"""You are correcting OCR output from handwritten {context}.

The OCR may have errors in:
- Spelling
- Chemical formulas (preserve subscripts as numbers: H2O not H₂O)
- Mathematical symbols
- Formatting

OCR Output:
{ocr_text}

Instructions:
1. Fix spelling errors
2. Preserve ALL chemical formulas (H2O, CaCl2, ΔH, etc.)
3. Preserve mathematical symbols and equations
4. Fix obvious OCR mistakes
5. Keep technical terms accurate
6. Return ONLY the corrected text - NO explanations, NO made-up content
7. If text mentions diagrams or boxes, keep those references

Corrected text:"""

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system", 
                        "content": "You correct OCR errors in scientific notes. Return ONLY the corrected text. Never add content that wasn't in the original. Preserve all technical accuracy."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Lower temperature for less creativity
                max_tokens=4000
            )
            
            corrected = completion.choices[0].message.content.strip()
            
            # Validate that we got actual correction, not hallucination
            if "please provide" in corrected.lower() or "i'd be happy" in corrected.lower():
                print("⚠️ LLM hallucinated, returning original OCR")
                return ocr_text
            
            print(f"✅ Text corrected by Llama 3.3")
            return corrected
            
        except Exception as e:
            print(f"❌ Groq error: {e}")
            return ocr_text
    
    def structure_content(self, text_blocks):
        """
        Use LLM to intelligently structure the content
        """
        if not self.client:
            return text_blocks
        
        # Handle both list and string input
        if isinstance(text_blocks, list):
            combined_text = "\n\n".join([
                block.get('text', '') if isinstance(block, dict) else str(block) 
                for block in text_blocks 
                if block
            ])
        else:
            combined_text = str(text_blocks)
        
        # Don't process if empty
        if not combined_text or len(combined_text.strip()) < 10:
            return combined_text
        
        prompt = f"""Structure these handwritten chemistry notes properly.

Original Notes:
{combined_text}

Instructions:
1. Add markdown headings where appropriate (# for main, ## for sub)
2. Keep all content from original - don't add or remove
3. Organize into logical sections
4. Preserve all equations, formulas, and diagrams
5. Use bullet points for lists
6. Keep [DIAGRAM] markers

Return structured markdown:"""

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You structure scientific notes into readable markdown. Keep all original content. Don't add or invent content."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            structured = completion.choices[0].message.content.strip()
            
            # Validate
            if "please provide" in structured.lower():
                print("⚠️ LLM couldn't structure, returning original")
                return combined_text
            
            return structured
            
        except Exception as e:
            print(f"❌ Structuring error: {e}")
            return combined_text