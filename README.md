# 📝 Handwriting to Word Converter

AI-powered tool to convert handwritten notes into editable Word documents using hybrid OCR (Groq Llama Vision API + Florence-2 + GOT-OCR 2.0 + EasyOCR) with intelligent diagram detection.

## ✨ Features

- **🤖 Hybrid OCR System**: Intelligent fallback from API → Local GPU → CPU
  - Groq Llama Vision API (fast, high quality)
  - Florence-2 Local (offline, mixed content)
  - GOT-OCR 2.0 (handwriting + formulas, LaTeX output)
  - EasyOCR (universal fallback)
- **📊 Smart Diagram Detection**: Automatically detects and extracts diagrams/graphs
- **✨ LLM Post-Processing**: Fix OCR errors and improve text quality
- **📄 Professional Word Documents**: Proper formatting with embedded diagrams
- **🎨 Beautiful Web UI**: Streamlit-based interface with real-time progress
- **📈 Detailed Metrics**: See which OCR method was used and confidence scores
- **🔄 Model Caching**: Keeps models loaded for faster subsequent processing

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/handwriting-to-word.git
cd handwriting-to-word

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key from: https://console.groq.com/

### 3. Run the Application

#### Option A: Web Interface (Recommended)
```bash
streamlit run app.py
```

Or double-click `run_app.bat` on Windows.

#### Option B: Command Line
```bash
# API-first mode (default)
python -m src.main_pipeline uploads/image.jpg MyNotes

# Local-only mode (skip API)
python -m src.main_pipeline uploads/image.jpg MyNotes --local

# Disable diagram detection
python -m src.main_pipeline uploads/image.jpg MyNotes --no-diagrams
```

## 📖 Usage

### Web Interface

1. **Upload Image(s)**: Drag & drop or click to upload handwritten notes
2. **Choose OCR Model**:
   - **Auto**: Smart fallback (API → Florence → GOT-OCR → EasyOCR)
   - **Llama Vision API**: Best quality, requires API key
   - **Florence-2 Local**: Mixed content, 6GB VRAM
   - **GOT-OCR 2.0 Local**: Best for handwriting + formulas
   - **EasyOCR**: Lightweight fallback
3. **Configure Options**:
   - Enable/disable diagram detection
   - Enable/disable LLM correction
   - Show text preview and detected diagrams
4. **Convert**: Click "Convert to Word Document"
5. **Download**: Get your Word document with embedded diagrams

### Command Line Examples

```bash
# Basic usage with auto model selection
python -m src.main_pipeline uploads/chemistry_notes.jpg ChemistryNotes

# Force local models only
python -m src.main_pipeline uploads/math_hw.png MathHomework --local

# Disable diagram detection for faster processing
python -m src.main_pipeline uploads/text_only.jpg TextNotes --no-diagrams
```

## 📁 Project Structure

```
handwriting_to_word/
├── app.py                          # Streamlit web interface
├── requirements.txt                # Python dependencies
├── .env                           # API keys (create this)
├── src/
│   ├── main_pipeline.py           # Main pipeline orchestrator
│   ├── preprocessing/
│   │   ├── image_processor.py     # Image preprocessing
│   │   └── diagram_detector.py    # Diagram detection & extraction
│   ├── ocr/
│   │   ├── hybrid_ocr.py          # Hybrid OCR system
│   │   ├── vision_ocr.py          # Groq Llama Vision API
│   │   ├── florence_local_ocr.py  # Florence-2 local model
│   │   ├── got_ocr_local.py       # GOT-OCR 2.0 local model
│   │   └── ocr_engine.py          # Legacy OCR engine
│   ├── postprocessing/
│   │   └── llm_corrector.py       # LLM text correction
│   └── document_generation/
│       └── word_generator.py      # Word document creation
├── uploads/                        # Upload images here
├── outputs/                        # Generated Word documents
└── temp/                          # Temporary processing files
```

## 🔧 Configuration

### OCR Models

**Auto Mode (Default)**
- Tries Groq Llama Vision API first
- Falls back to Florence-2 if API fails
- Uses GOT-OCR 2.0 for handwriting-heavy content
- Uses EasyOCR as last resort

**Local Only Mode**
- Skips API, uses local models directly
- Requires CUDA GPU with 4-8GB VRAM (recommended)
- Works on CPU (slower)

### Model Specifications

| Model | Size | VRAM | Best For | Confidence |
|-------|------|------|----------|------------|
| **Llama Vision API** | API | 0GB | All content types | ~95% |
| **Florence-2** | 230M | 4GB | Mixed content + diagrams | ~85% |
| **GOT-OCR 2.0** | 580M | 6GB | Handwriting + formulas | ~90% |
| **EasyOCR** | 50M | 2GB | Fallback | Per-char |

### LLM Correction

- Uses Groq API with Llama models
- Fixes OCR errors and improves formatting
- Structures content intelligently
- Can be disabled for faster processing

## 📊 Diagram Detection

The system automatically detects and handles diagrams using computer vision:

1. **Edge Detection**: Finds diagram boundaries using Canny edge detection
2. **Contour Analysis**: Identifies enclosed regions and shapes
3. **Heuristic Filtering**: Classifies regions as diagrams based on:
   - Straight lines (graph axes, borders)
   - Enclosed shapes (boxes, circles, nodes)
   - Size and aspect ratio
4. **Text Masking**: Creates text-only version for better OCR
5. **Document Embedding**: Adds diagrams to Word document with captions


## 🐛 Troubleshooting

### "No module named 'preprocessing'"
```bash
# Run from project root with -m flag
python -m src.main_pipeline uploads/image.jpg OutputName
```

### "All OCR methods failed"
- Check image quality and readability
- Verify API key if using API mode
- Ensure GPU is available for local mode
- Try preprocessing the image first

### Out of Memory Errors
- Use CPU mode (slower): Set device='cpu' in code
- Reduce image resolution before processing
- Close other GPU applications
- Try EasyOCR only mode

### API Rate Limits
- Switch to Local Only mode in settings
- Wait and retry
- Check Groq API quota at console.groq.com

### Model Download Issues
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('stepfun-ai/GOT-OCR2_0', trust_remote_code=True)"
```

## 🛠️ Development

### Running Tests
```bash
python test_pipeline.py
```

### Adding New OCR Engines
1. Create new class in `src/ocr/`
2. Implement `extract_text_from_image()` method
3. Add to `HybridOCR` fallback chain in `hybrid_ocr.py`

### Customizing Word Output
Edit `src/document_generation/word_generator.py` to change:
- Fonts and styling
- Heading detection rules
- Diagram embedding behavior

### Model Caching
Models are cached in session state for the web interface:
- Faster subsequent processing
- Memory efficient
- Automatic cleanup on model change

## 📝 Recent Updates

- ✅ **GOT-OCR 2.0 Integration**: Replaced TrOCR with state-of-the-art handwriting model
- ✅ **Enhanced Diagram Detection**: Improved accuracy and embedding
- ✅ **Model Caching**: Faster processing with session-based model reuse
- ✅ **Unique File Handling**: Timestamp-based uploads prevent conflicts
- ✅ **Better Error Handling**: Graceful fallbacks and detailed logging
- ✅ **Formula Support**: LaTeX output for mathematical expressions

## 📄 License

This project is for educational purposes. Please respect the licenses of the underlying models:
- Florence-2: MIT License
- GOT-OCR 2.0: Apache 2.0 License
- EasyOCR: Apache 2.0 License

## 🙏 Acknowledgments

- **Florence-2**: Microsoft's vision foundation model
- **GOT-OCR 2.0**: StepFun AI's advanced OCR model
- **Groq**: Fast LLM inference API
- **EasyOCR**: Reliable OCR library
- **Streamlit**: Beautiful web interface framework

## 📧 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments and documentation
3. Test with different images and settings
4. Verify API keys and dependencies

---

Made with ❤️ using Python, PyTorch, and AI | Perfect for converting chemistry notes, math homework, and handwritten documents!