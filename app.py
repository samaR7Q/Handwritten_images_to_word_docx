import streamlit as st
import os
import sys
from pathlib import Path
from PIL import Image
import time
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.image_processor import ImagePreprocessor
from src.preprocessing.diagram_detector import DiagramDetector
from src.ocr.hybrid_ocr import HybridOCR
from src.postprocessing.llm_corrector import LLMCorrector
from src.document_generation.word_generator import WordGenerator

# Page config
st.set_page_config(
    page_title="Handwriting to Word Converter",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (keep your existing CSS)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'ocr_info' not in st.session_state:
    st.session_state.ocr_info = {}
if 'diagram_info' not in st.session_state:
    st.session_state.diagram_info = {}
if 'ocr_engine' not in st.session_state:
    st.session_state.ocr_engine = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Header
st.markdown('<h1 class="main-header">📝 Handwriting to Word Converter</h1>', unsafe_allow_html=True)
st.markdown("### Transform your handwritten notes into editable Word documents with AI")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OCR Model Selection
    st.subheader("🤖 OCR Model")
    ocr_model = st.selectbox(
        "Select OCR Engine:",
        [
            "Auto (Smart Fallback)",
            "Llama Vision API (Groq)",
            "Florence-2 Local",
            "GOT-OCR 2.0 Local",
            "EasyOCR"
        ],
        help="Auto: Tries API → Florence → GOT-OCR → EasyOCR"
    )
    
    # Map selection to parameters
    if ocr_model == "Llama Vision API (Groq)":
        prefer_local = False
        force_model = "api"
    elif ocr_model == "Florence-2 Local":
        prefer_local = True
        force_model = "florence"
    elif ocr_model == "GOT-OCR 2.0 Local":
        prefer_local = True
        force_model = "got"
    elif ocr_model == "EasyOCR":
        prefer_local = True
        force_model = "easyocr"
    else:  # Auto
        prefer_local = False
        force_model = None
    
    # Diagram handling
    st.subheader("📊 Diagram Detection")
    detect_diagrams = st.checkbox(
        "Enable Diagram Detection",
        value=True,
        help="Automatically detect and extract diagrams/graphs"
    )
    
    if detect_diagrams:
        diagram_handling = st.radio(
            "Diagram Handling:",
            ["Embed in Document", "Describe with AI", "Both"],
            help="How to handle detected diagrams"
        )
    
    # LLM Correction
    st.subheader("✨ Text Enhancement")
    use_llm = st.checkbox(
        "Enable LLM Correction",
        value=True,
        help="Use AI to fix OCR errors"
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        try_preprocessed = st.checkbox("Retry with Preprocessed", value=True)
        show_preview = st.checkbox("Show Text Preview", value=True)
        show_diagrams = st.checkbox("Show Detected Diagrams", value=True)
    
    st.divider()
    
    # Model info
    st.subheader("ℹ️ Model Information")
    if ocr_model == "Florence-2 Local":
        st.info("""
        **Florence-2-Base**
        - 230M parameters
        - Best for: Mixed content
        - Can describe diagrams
        - Confidence: Estimated (~85%)
        """)
    elif ocr_model == "GOT-OCR 2.0 Local":
        st.info("""
        **GOT-OCR 2.0**
        - 580M parameters
        - Best for: Handwriting + Formulas
        - Supports LaTeX formulas
        - Format preservation
        - Confidence: Estimated (~90%)
        """)
    elif ocr_model == "Llama Vision API (Groq)":
        st.info("""
        **Llama 4 Scout Vision**
        - API-based (requires key)
        - Best quality overall
        - Handles all content types
        - Confidence: Estimated (~95%)
        """)
    elif ocr_model == "EasyOCR":
        st.info("""
        **EasyOCR**
        - Lightweight
        - Reliable fallback
        - Confidence: Per-character
        """)
    
    # API Status
    st.subheader("🔑 API Status")
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        st.success("✅ Groq API Key Found")
    else:
        st.warning("⚠️ No Groq API Key")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Images")
    
    uploaded_files = st.file_uploader(
        "Choose image(s) of handwritten notes",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} image(s) uploaded")
        
        # Show thumbnails
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, uploaded_file in enumerate(uploaded_files[:3]):
            with cols[idx]:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True, caption=uploaded_file.name)

with col2:
    st.subheader("💾 Output Settings")
    
    output_name = st.text_input(
        "Document Name",
        value="MyNotes",
        help="Name for output Word document"
    )

# Process button
st.divider()

if uploaded_files:
    if st.button("🚀 Convert to Word Document", type="primary"):
        st.session_state.processed = False
        
        # Progress tracking (setup first)
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.expander("📋 Processing Log", expanded=True)
        
        # Save uploaded files
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        # Save uploaded files with timestamp to avoid conflicts
        import time
        timestamp = int(time.time())
        
        image_paths = []
        for idx, uploaded_file in enumerate(uploaded_files):
            # Use timestamp to ensure unique filenames
            file_ext = os.path.splitext(uploaded_file.name)[1]
            file_path = f"uploads/{timestamp}_{idx}_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(file_path)
            
        with log_container:
            st.write(f"📁 Saved {len(image_paths)} file(s):")
            for path in image_paths:
                st.write(f"   - {os.path.basename(path)}")
        
        try:
            # Initialize pipeline
            with log_container:
                st.write("🔧 Initializing pipeline...")
            status_text.text("🔧 Initializing pipeline...")
            progress_bar.progress(10)
            
            preprocessor = ImagePreprocessor()
            diagram_detector = DiagramDetector() if detect_diagrams else None
            
            # Reuse OCR engine if same model, otherwise create new one
            model_key = f"{prefer_local}_{force_model if force_model else 'auto'}"
            if st.session_state.current_model != model_key or st.session_state.ocr_engine is None:
                with log_container:
                    st.write(f"📥 Loading OCR model: {ocr_model}...")
                if st.session_state.ocr_engine:
                    st.session_state.ocr_engine.cleanup()
                st.session_state.ocr_engine = HybridOCR(prefer_local=prefer_local, local_model=force_model if force_model else "auto")
                st.session_state.current_model = model_key
            else:
                with log_container:
                    st.write(f"♻️ Reusing cached OCR model: {ocr_model}")
            
            ocr_engine = st.session_state.ocr_engine
            llm_corrector = LLMCorrector() if use_llm else None
            word_generator = WordGenerator()
            
            progress_bar.progress(20)
            
            # Process all uploaded images (or just first if multiple)
            if len(image_paths) > 1:
                with log_container:
                    st.write(f"📚 Processing {len(image_paths)} images...")
                    st.write("   (Currently processing first image only)")
            
            img_path = image_paths[0]
            
            with log_container:
                st.write(f"📸 Processing: {os.path.basename(img_path)}")
            status_text.text(f"📸 Processing {os.path.basename(img_path)}...")
            
            # Step 1: Preprocessing
            with log_container:
                st.write("STEP 1: Preprocessing...")
            preprocessed_img, preprocessed_path = preprocessor.preprocess(img_path)
            progress_bar.progress(30)
            
            # Step 2: Diagram detection
            diagrams = []
            if detect_diagrams and diagram_detector:
                with log_container:
                    st.write("STEP 2: Detecting diagrams...")
                status_text.text("🔍 Detecting diagrams...")
                diagram_result = diagram_detector.detect_and_extract(img_path)
                
                if diagram_result.get('has_diagrams'):
                    diagrams = diagram_result.get('diagram_regions', [])
                    with log_container:
                        st.write(f"   Found {len(diagrams)} diagram(s)")
                    st.session_state.diagram_info = {'count': len(diagrams)}
                    
                    if show_diagrams:
                        with log_container:
                            st.write("**Detected Diagrams:**")
                            cols = st.columns(min(len(diagrams), 3))
                            for idx, diag in enumerate(diagrams[:3]):
                                with cols[idx]:
                                    diag_img = cv2.imread(diag['path'])
                                    diag_img_rgb = cv2.cvtColor(diag_img, cv2.COLOR_BGR2RGB)
                                    st.image(diag_img_rgb, caption=f"Diagram {idx+1}", use_column_width=True)
            
            progress_bar.progress(40)
            
            # Step 3: OCR
            with log_container:
                st.write("STEP 3: Running OCR...")
            status_text.text("🔍 Running OCR...")
            
            # Try original first
            ocr_result = ocr_engine.extract_text_from_image(img_path)
            
            # If failed and retry enabled, try preprocessed
            if try_preprocessed and (not ocr_result.get('text') or len(ocr_result['text']) < 50):
                with log_container:
                    st.write("🔄 Retrying with preprocessed image...")
                ocr_result = ocr_engine.extract_text_from_image(preprocessed_path)
            
            extracted_text = ocr_result.get('text', '')
            confidence = ocr_result.get('confidence', 0)
            method = ocr_result.get('method', 'unknown')
            
            progress_bar.progress(60)
            
            # Store OCR info
            st.session_state.ocr_info = {
                'method': method,
                'confidence': confidence,
                'text_length': len(extracted_text)
            }
            
            with log_container:
                st.write(f"📊 OCR Results:")
                st.write(f"   - Method: **{method}**")
                st.write(f"   - Confidence: **{confidence:.2%}**")
                st.write(f"   - Text length: **{len(extracted_text)}** characters")
            
            if len(extracted_text) < 20:
                st.error("❌ OCR failed to extract meaningful text!")
                st.stop()
            
            # Show preview
            if show_preview:
                with log_container:
                    st.text_area("📄 Extracted Text Preview:", extracted_text[:500] + "...", height=150)
            
            # Step 4: LLM correction
            if use_llm and llm_corrector:
                with log_container:
                    st.write("STEP 4: LLM post-processing...")
                status_text.text("✨ Enhancing text with AI...")
                
                corrected_text = llm_corrector.correct_text(extracted_text)
                structured_text = llm_corrector.structure_content(corrected_text)
                progress_bar.progress(80)
            else:
                structured_text = extracted_text
                progress_bar.progress(80)
            
            # Step 5: Generate Word document
            with log_container:
                st.write("STEP 5: Generating Word document...")
            status_text.text("📄 Creating Word document...")
            
            # Pass diagrams to word generator if available
            diagrams_to_embed = None
            if detect_diagrams and diagrams:
                diagrams_to_embed = diagrams
                with log_container:
                    st.write(f"📊 Embedding {len(diagrams)} diagram(s) in document")
            
            output_path = word_generator.create_document(
                structured_text, 
                output_name,
                diagrams=diagrams_to_embed
            )
            progress_bar.progress(100)
            
            # Success
            st.session_state.processed = True
            st.session_state.output_path = output_path
            status_text.empty()
            
            with log_container:
                st.success(f"✅ SUCCESS! Document saved to: {output_path}")
            
            time.sleep(0.5)
            st.balloons()
            
            # Don't cleanup - keep models cached for next run
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with log_container:
                st.exception(e)

# Results section
if st.session_state.processed and st.session_state.output_path:
    st.divider()
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("### ✅ Conversion Complete!")
    
    # Show metrics
    if st.session_state.ocr_info:
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("OCR Method", st.session_state.ocr_info.get('method', 'N/A'))
        with col_info2:
            st.metric("Confidence", f"{st.session_state.ocr_info.get('confidence', 0):.1%}")
        with col_info3:
            st.metric("Text Length", f"{st.session_state.ocr_info.get('text_length', 0)} chars")
    
    # Show diagram info
    if st.session_state.diagram_info:
        st.info(f"📊 Detected {st.session_state.diagram_info.get('count', 0)} diagram(s)")
    
    st.markdown(f"**Document saved:** `{st.session_state.output_path}`")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download button
    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        if os.path.exists(st.session_state.output_path):
            with open(st.session_state.output_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Word Document",
                    data=file,
                    file_name=f"{output_name}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Made with ❤️ using Streamlit | Powered by Groq Llama Vision & Florence-2</p>
</div>
""", unsafe_allow_html=True)