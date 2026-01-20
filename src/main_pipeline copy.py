import os
import sys
from .preprocessing.image_processor import ImagePreprocessor
from .ocr.vision_ocr import VisionOCR 
from .postprocessing.llm_corrector import LLMCorrector
from .document_generation.word_generator import WordGenerator

class HandwritingToWordPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, use_vision=True):
        print("🚀 Initializing pipeline...")
        self.preprocessor = ImagePreprocessor()
        
        # Use Vision-based OCR for better accuracy
        if use_vision:
            self.ocr_engine = VisionOCR()
        else:
            from ocr.ocr_engine import OCREngine
            self.ocr_engine = OCREngine()
        
        self.llm_corrector = LLMCorrector()
        self.word_generator = WordGenerator()
        print("✅ Pipeline ready!")
    
    def process_image(self, image_path, output_name="converted_notes"):
        """
        Full pipeline: Image → OCR → LLM Correction → Word Doc
        """
        print(f"\n{'='*60}")
        print(f"📄 Processing: {image_path}")
        print(f"{'='*60}\n")
        
        # Step 1: Light preprocessing (just resize if needed)
        print("STEP 1: Preprocessing...")
        preprocessed_img, preprocessed_path = self.preprocessor.preprocess(image_path)
        
        # Step 2: Vision-based OCR (much better for handwriting)
        print("\nSTEP 2: Running Vision OCR...")
        
        # Use original image for vision models (they handle preprocessing internally)
        ocr_result = self.ocr_engine.extract_text_from_image(image_path)
        
        if not ocr_result.get('text') or len(ocr_result['text']) < 50:
            print("⚠️ OCR returned very little text, trying preprocessed image...")
            ocr_result = self.ocr_engine.extract_text_from_image(preprocessed_path)
        
        extracted_text = ocr_result.get('text', '')
        confidence = ocr_result.get('confidence', 0)
        
        print(f"\n📊 OCR Stats:")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Text length: {len(extracted_text)} characters")
        print(f"   Preview:\n   {extracted_text[:200]}...")
        
        if len(extracted_text) < 20:
            print("\n❌ ERROR: OCR failed to extract meaningful text!")
            print("   Suggestions:")
            print("   1. Check image quality")
            print("   2. Ensure API keys are configured")
            print("   3. Try different preprocessing settings")
            return None
        
        # Step 3: LLM correction (optional if Vision OCR is good)
        print("\nSTEP 3: LLM post-processing...")
        corrected_text = self.llm_corrector.correct_text(extracted_text)
        structured_text = self.llm_corrector.structure_content(corrected_text)
        
        # Step 4: Generate Word document
        print("\nSTEP 4: Generating Word document...")
        output_path = self.word_generator.create_document(structured_text, output_name)
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! Document saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python main_pipeline.py <image_path> [output_name]")
        print("Example: python main_pipeline.py uploads/notes1.jpg MyNotes")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else "converted_notes"
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Run pipeline with Vision OCR
    pipeline = HandwritingToWordPipeline(use_vision=True)
    pipeline.process_image(image_path, output_name)


if __name__ == "__main__":
    main()