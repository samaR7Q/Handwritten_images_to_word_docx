import streamlit as st
import os
import sys
from pathlib import Path
from PIL import Image
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocessing.image_processor import ImagePreprocessor
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

# Custom CSS
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
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
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

# Header
st.markdown('<h1 class="main-header">📝 Handwriting to Word Converter</h1>', unsafe_allow_html=True)
st.markdown("### Transform your handwritten notes into editable Word documents with AI")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("🤖 OCR Strategy")
    ocr_mode = st.radio(
        "Select OCR Mode:",
        ["API First (Groq → Local → EasyOCR)", "Local Only (Florence-2 → EasyOCR)"],
        help="API First: Try Groq Llama Vision API, fallback to local. Local Only: Skip API, use Florence-2 directly."
    )
    prefer_local = "Local Only" in ocr_mode
    
    # LLM Correction
    st.subheader("✨ Text Enhancement")
    use_llm = st.checkbox(
        "Enable LLM Correction",
        value=True,
        help="Use AI to fix OCR errors and improve text quality"
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        try_preprocessed = st.checkbox(
            "Retry with Preprocessed Image", 
            value=True,
            help="If OCR fails on original, try preprocessed version"
        )
        show_preview = st.checkbox("Show Text Preview", value=True)
    
    st.divider()
    
    # Info
    st.info("""
    **💡 Tips:**
    - Use clear, well-lit images
    - Avoid shadows and glare
    - Higher resolution = better results
    - API mode requires Groq API key
    - Local mode needs ~6GB VRAM
    """)
    
    # API Status
    st.subheader("🔑 API Status")
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        st.success("✅ Groq API Key Found")
    else:
        st.warning("⚠️ No Groq API Key")
        st.caption("Add GROQ_API_KEY to .env for API mode")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Images")
    
    uploaded_files = st.file_uploader(
        "Choose image(s) of handwritten notes",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="You can upload multiple images to process separately"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} image(s) uploaded")
        
        # Show thumbnails
        st.write("**Preview:**")
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, uploaded_file in enumerate(uploaded_files[:3]):
            with cols[idx]:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True, caption=uploaded_file.name)
        
        if len(uploaded_files) > 3:
            st.info(f"+ {len(uploaded_files) - 3} more image(s)")

with col2:
    st.subheader("💾 Output Settings")
    
    output_name = st.text_input(
        "Document Name",
        value="MyNotes",
        help="Name for the output Word document (without extension)"
    )
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("temp", exist_ok=True)

# Process button
st.divider()

if uploaded_files:
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        process_btn = st.button(
            "🚀 Convert to Word Document",
            type="primary"
        )
    
    if process_btn:
        st.session_state.processed = False
        
        # Save uploaded files
        image_paths = []
        for uploaded_file in uploaded_files:
            file_path = f"uploads/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(file_path)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.expander("📋 Processing Log", expanded=True)
        
        try:
            # Initialize pipeline
            with log_container:
                st.write("🔧 Initializing pipeline...")
            status_text.text("🔧 Initializing pipeline...")
            progress_bar.progress(10)
            
            preprocessor = ImagePreprocessor()
            ocr_engine = HybridOCR(prefer_local=prefer_local)
            llm_corrector = LLMCorrector() if use_llm else None
            word_generator = WordGenerator()
            
            progress_bar.progress(20)
            
            # Process first image (or combine multiple)
            # For simplicity, processing first image. You can extend for multiple.
            img_path = image_paths[0]
            
            with log_container:
                st.write(f"📸 Processing: {os.path.basename(img_path)}")
            status_text.text(f"📸 Processing image...")
            
            # Step 1: Preprocessing
            with log_container:
                st.write("STEP 1: Preprocessing...")
            preprocessed_img, preprocessed_path = preprocessor.preprocess(img_path)
            progress_bar.progress(30)
            
            # Step 2: OCR
            with log_container:
                st.write("STEP 2: Running Hybrid OCR...")
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
            
            # Step 3: LLM correction
            if use_llm and llm_corrector:
                with log_container:
                    st.write("STEP 3: LLM post-processing...")
                status_text.text("✨ Enhancing text with AI...")
                
                corrected_text = llm_corrector.correct_text(extracted_text)
                structured_text = llm_corrector.structure_content(corrected_text)
                progress_bar.progress(80)
            else:
                structured_text = extracted_text
                progress_bar.progress(80)
            
            # Step 4: Generate Word document
            with log_container:
                st.write("STEP 4: Generating Word document...")
            status_text.text("📄 Creating Word document...")
            
            output_path = word_generator.create_document(structured_text, output_name)
            progress_bar.progress(100)
            
            # Success
            st.session_state.processed = True
            st.session_state.output_path = output_path
            status_text.empty()
            
            with log_container:
                st.success(f"✅ SUCCESS! Document saved to: {output_path}")
            
            time.sleep(0.5)
            st.balloons()
            
            # Cleanup
            ocr_engine.cleanup()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with log_container:
                st.exception(e)

# Results section
if st.session_state.processed and st.session_state.output_path:
    st.divider()
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("### ✅ Conversion Complete!")
    
    # Show OCR info
    if st.session_state.ocr_info:
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("OCR Method", st.session_state.ocr_info.get('method', 'N/A'))
        with col_info2:
            st.metric("Confidence", f"{st.session_state.ocr_info.get('confidence', 0):.1%}")
        with col_info3:
            st.metric("Text Length", f"{st.session_state.ocr_info.get('text_length', 0)} chars")
    
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
