import os
import sys
from .preprocessing.image_processor import ImagePreprocessor
from .ocr.hybrid_ocr import HybridOCR  # NEW: Hybrid system
from .postprocessing.llm_corrector import LLMCorrector
from .document_generation.word_generator import WordGenerator

class HandwritingToWordPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, prefer_local=False):
        """
        Args:
            prefer_local: If True, skip API and use local model directly
        """
        print("🚀 Initializing pipeline...")
        print(f"   Mode: {'LOCAL-FIRST' if prefer_local else 'API-FIRST (with local fallback)'}")
        
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = HybridOCR(prefer_local=prefer_local)  # Hybrid OCR
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
        
        # Step 1: Preprocessing (optional for Florence/Vision models)
        print("STEP 1: Preprocessing...")
        preprocessed_img, preprocessed_path = self.preprocessor.preprocess(image_path)
        
        # Step 2: Hybrid OCR (API → Local → Fallback)
        print("\nSTEP 2: Running OCR...")
        
        # Try original image first (vision models prefer original)
        ocr_result = self.ocr_engine.extract_text_from_image(image_path)
        
        # If failed, try preprocessed
        if not ocr_result.get('text') or len(ocr_result['text']) < 50:
            print("  🔄 Retrying with preprocessed image...")
            ocr_result = self.ocr_engine.extract_text_from_image(preprocessed_path)
        
        extracted_text = ocr_result.get('text', '')
        confidence = ocr_result.get('confidence', 0)
        method = ocr_result.get('method', 'unknown')
        
        print(f"\n📊 OCR Results:")
        print(f"   Method: {method}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Text length: {len(extracted_text)} characters")
        print(f"   Preview:\n   {extracted_text[:200]}...")
        
        if len(extracted_text) < 20:
            print("\n❌ ERROR: OCR failed to extract meaningful text!")
            print("   All methods failed. Please check:")
            print("   1. Image quality and readability")
            print("   2. API keys if using API mode")
            print("   3. GPU availability for local model")
            return None
        
        # Step 3: LLM post-processing
        print("\nSTEP 3: LLM post-processing...")
        corrected_text = self.llm_corrector.correct_text(extracted_text)
        structured_text = self.llm_corrector.structure_content(corrected_text)
        
        # Step 4: Generate Word document
        print("\nSTEP 4: Generating Word document...")
        output_path = self.word_generator.create_document(structured_text, output_name)
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! Document saved to: {output_path}")
        print(f"   OCR Method Used: {method}")
        print(f"{'='*60}\n")
        
        return output_path
    
    def cleanup(self):
        """Clean up resources"""
        self.ocr_engine.cleanup()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert handwritten notes to Word documents')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('output_name', nargs='?', default='converted_notes', 
                       help='Output document name (default: converted_notes)')
    parser.add_argument('--local', action='store_true', 
                       help='Use local model only (skip API)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = HandwritingToWordPipeline(prefer_local=args.local)
    
    try:
        result = pipeline.process_image(args.image_path, args.output_name)
        if not result:
            sys.exit(1)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()