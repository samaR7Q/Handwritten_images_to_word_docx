import os
import sys
from .preprocessing.image_processor import ImagePreprocessor
from .preprocessing.diagram_detector import DiagramDetector
from .ocr.hybrid_ocr import HybridOCR
from .postprocessing.llm_corrector import LLMCorrector
from .document_generation.word_generator import WordGenerator

class HandwritingToWordPipeline:
    """Main pipeline orchestrator with diagram detection"""
    
    def __init__(self, prefer_local=False, detect_diagrams=True):
        """
        Args:
            prefer_local: If True, skip API and use local model directly
            detect_diagrams: If True, detect and extract diagrams
        """
        print("🚀 Initializing pipeline...")
        print(f"   Mode: {'LOCAL-FIRST' if prefer_local else 'API-FIRST (with local fallback)'}")
        print(f"   Diagram Detection: {'ENABLED' if detect_diagrams else 'DISABLED'}")
        
        self.preprocessor = ImagePreprocessor()
        self.diagram_detector = DiagramDetector() if detect_diagrams else None
        self.ocr_engine = HybridOCR(prefer_local=prefer_local)
        self.llm_corrector = LLMCorrector()
        self.word_generator = WordGenerator()
        
        print("✅ Pipeline ready!")
    
    def process_image(self, image_path, output_name="converted_notes"):
        """
        Full pipeline: Image → Diagram Detection → OCR → LLM Correction → Word Doc
        """
        print(f"\n{'='*60}")
        print(f"📄 Processing: {image_path}")
        print(f"{'='*60}\n")
        
        # Step 1: Preprocessing
        print("STEP 1: Preprocessing...")
        preprocessed_img, preprocessed_path = self.preprocessor.preprocess(image_path)
        
        # Step 2: Diagram Detection (optional)
        diagram_info = None
        text_only_path = image_path
        
        if self.diagram_detector:
            print("\nSTEP 2: Detecting diagrams...")
            diagram_info = self.diagram_detector.detect_and_extract(image_path)
            
            if diagram_info.get('has_diagrams'):
                diagrams = diagram_info.get('diagram_regions', [])
                print(f"   ✅ Found {len(diagrams)} diagram(s)")
                for i, diag in enumerate(diagrams):
                    print(f"      Diagram {i+1}: {diag['path']}")
                
                # Use text-only version for OCR
                text_only_path = diagram_info.get('text_only_image', image_path)
            else:
                print("   No diagrams detected")
        
        # Step 3: Hybrid OCR
        print(f"\nSTEP {'3' if self.diagram_detector else '2'}: Running OCR...")
        
        # Try original/text-only image first
        ocr_result = self.ocr_engine.extract_text_from_image(text_only_path)
        
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
        
        # Step 4: LLM post-processing
        print(f"\nSTEP {'4' if self.diagram_detector else '3'}: LLM post-processing...")
        corrected_text = self.llm_corrector.correct_text(extracted_text)
        structured_text = self.llm_corrector.structure_content(corrected_text)
        
        # Step 5: Generate Word document
        print(f"\nSTEP {'5' if self.diagram_detector else '4'}: Generating Word document...")
        
        # Pass diagrams to word generator if available
        diagrams_to_embed = None
        if diagram_info and diagram_info.get('has_diagrams'):
            diagrams_to_embed = diagram_info.get('diagram_regions', [])
        
        output_path = self.word_generator.create_document(
            structured_text, 
            output_name,
            diagrams=diagrams_to_embed
        )
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! Document saved to: {output_path}")
        print(f"   OCR Method Used: {method}")
        if diagram_info and diagram_info.get('has_diagrams'):
            print(f"   Diagrams Detected: {len(diagram_info.get('diagram_regions', []))}")
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
    parser.add_argument('--no-diagrams', action='store_true',
                       help='Disable diagram detection')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = HandwritingToWordPipeline(
        prefer_local=args.local,
        detect_diagrams=not args.no_diagrams
    )
    
    try:
        result = pipeline.process_image(args.image_path, args.output_name)
        if not result:
            sys.exit(1)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()