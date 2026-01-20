# 📝 Handwriting to Word Converter

AI-powered tool to convert handwritten notes into editable Word documents using hybrid OCR (Groq Llama Vision API + Florence-2 + EasyOCR).

## ✨ Features

- **🤖 Hybrid OCR System**: Intelligent fallback from API → Local GPU → CPU
  - Groq Llama Vision API (fast, high quality)
  - Florence-2 Local (offline, 6GB VRAM)
  - EasyOCR (universal fallback)
- **✨ LLM Post-Processing**: Fix OCR errors and improve text quality
- **📄 Word Document Generation**: Professional formatting with proper structure
- **🎨 Beautiful Web UI**: Streamlit-based interface with real-time progress
- **📊 Detailed Metrics**: See which OCR method was used and confidence scores

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd handwriting_to_word

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
```

## 📖 Usage

### Web Interface

1. **Upload Image(s)**: Drag & drop or click to upload handwritten notes
2. **Choose OCR Mode**:
   - **API First**: Try Groq API → Florence-2 → EasyOCR
   - **Local Only**: Skip API, use Florence-2 → EasyOCR
3. **Configure Options**:
   - Enable/disable LLM correction
   - Retry with preprocessed image
   - Show text preview
4. **Convert**: Click "Convert to Word Document"
5. **Download**: Get your Word document

### Command Line

```bash
# Basic usage
python -m src.main_pipeline path/to/image.jpg OutputName

# Use local model only (no API)
python -m src.main_pipeline path/to/image.jpg OutputName --local

# Examples
python -m src.main_pipeline uploads/chemistry_notes.jpg ChemistryNotes
python -m src.main_pipeline uploads/math_hw.png MathHomework --local
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
│   │   └── image_processor.py     # Image preprocessing
│   ├── ocr/
│   │   ├── hybrid_ocr.py          # Hybrid OCR system
│   │   ├── vision_ocr.py          # Groq Llama Vision API
│   │   ├── florence_local_ocr.py  # Florence-2 local model
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

### OCR Modes

**API First (Default)**
- Tries Groq Llama Vision API first
- Falls back to Florence-2 if API fails
- Uses EasyOCR as last resort
- Requires: `GROQ_API_KEY` in `.env`

**Local Only**
- Skips API, uses Florence-2 directly
- Falls back to EasyOCR if needed
- Requires: CUDA GPU with 6GB+ VRAM (recommended)
- Works on CPU (slower)

### LLM Correction

- Uses Groq API with Llama models
- Fixes OCR errors and improves formatting
- Structures content intelligently
- Can be disabled for faster processing

## 💡 Tips for Best Results

- **Image Quality**: Use clear, well-lit images
- **Resolution**: Higher resolution = better OCR accuracy
- **Contrast**: Ensure good contrast between text and background
- **Orientation**: Keep text horizontal and properly aligned
- **Lighting**: Avoid shadows and glare

## 🐛 Troubleshooting

### "No module named 'preprocessing'"
```bash
# Run from project root with -m flag
python -m src.main_pipeline uploads/image.jpg OutputName
```

### "Input type (float) and bias type (struct c10::Half) should be the same"
This is fixed in the latest version. Update your code or reinstall.

### "All OCR methods failed"
- Check image quality and readability
- Verify API key if using API mode
- Ensure GPU is available for local mode
- Try preprocessing the image first

### Florence-2 Out of Memory
- Reduce image resolution
- Use CPU mode (slower): Set `device='cpu'` in code
- Close other GPU applications

### API Rate Limits
- Switch to Local Only mode
- Wait and retry
- Check Groq API quota

## 📊 Performance

| Method | Speed | Quality | Requirements |
|--------|-------|---------|--------------|
| Groq API | ⚡⚡⚡ Fast | ⭐⭐⭐⭐⭐ Excellent | API Key |
| Florence-2 | ⚡⚡ Medium | ⭐⭐⭐⭐ Very Good | 6GB VRAM |
| EasyOCR | ⚡ Slow | ⭐⭐⭐ Good | CPU/GPU |

## 🔑 API Keys

### Groq API (Free Tier)
1. Sign up at https://console.groq.com/
2. Create an API key
3. Add to `.env`: `GROQ_API_KEY=your_key_here`
4. Free tier includes generous limits

## 🛠️ Development

### Running Tests
```bash
python test_pipeline.py
```

### Adding New OCR Engines
1. Create new class in `src/ocr/`
2. Implement `extract_text_from_image()` method
3. Add to `HybridOCR` fallback chain

### Customizing Word Output
Edit `src/document_generation/word_generator.py` to change:
- Fonts and styling
- Heading detection
- Formatting rules

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **Florence-2**: Microsoft's vision foundation model
- **Groq**: Fast LLM inference API
- **EasyOCR**: Reliable OCR library
- **Streamlit**: Beautiful web interface framework

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with different images
4. Verify API keys and dependencies

---

Made with ❤️ using Python, PyTorch, and AI
