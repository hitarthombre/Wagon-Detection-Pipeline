# OCR Comparison Tool Guide

## Overview

A Streamlit-based application for comparing three OCR engines:
- **TrOCR** - Transformer-based OCR (HuggingFace)
- **PaddleOCR 2.7.x** - Stable deep learning OCR
- **Tesseract** - Traditional OCR with preprocessing

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements_ocr_comparison.txt
```

### 2. Install Tesseract Binary (for Tesseract OCR)

**Windows:**
```bash
# Download and install from:
https://github.com/UB-Mannheim/tesseract/wiki

# Add to PATH or set in code:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 3. Verify Installation

```bash
python -c "import transformers; print('TrOCR: OK')"
python -c "import paddleocr; print('PaddleOCR: OK')"
python -c "import pytesseract; print('Tesseract: OK')"
```

## Usage

### Start the Application

```bash
streamlit run ocr_comparison_app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Select OCR Engine** (sidebar)
   - Choose one engine: TrOCR, PaddleOCR 2.7.x, or Tesseract
   - Configure engine-specific options

2. **Provide Image**
   - Upload image file
   - Use webcam
   - Select sample image

3. **Run OCR**
   - Click "Run OCR" button
   - View extracted text
   - See processing time and confidence
   - Visualize bounding boxes (if available)

## OCR Engine Comparison

### TrOCR (Transformer-based)

**Best for:**
- Handwritten text
- Complex layouts
- Artistic fonts
- Low-quality images

**Characteristics:**
- Speed: Moderate (2-5 seconds per image)
- Accuracy: Excellent for handwritten
- GPU: Highly recommended
- Bounding boxes: No
- Confidence scores: Not provided

**Installation:**
```bash
pip install transformers torch pillow
```

**Model Size:** ~300MB (downloads on first use)

---

### PaddleOCR 2.7.x (Stable)

**Best for:**
- Printed text
- Multi-language documents
- Production environments
- Batch processing

**Characteristics:**
- Speed: Fast (0.5-2 seconds per image)
- Accuracy: Excellent for printed text
- GPU: Optional (2-3x speedup)
- Bounding boxes: Yes
- Confidence scores: Yes (per text region)

**Installation:**
```bash
pip install paddleocr==2.7.0.3
```

**Important:** Use version 2.7.x, NOT 3.x (compatibility issues)

**Model Size:** ~10MB (downloads on first use)

---

### Tesseract OCR

**Best for:**
- Printed text
- Digits and numbers
- Simple layouts
- Fast processing

**Characteristics:**
- Speed: Very fast (0.1-0.5 seconds per image)
- Accuracy: Good with preprocessing
- GPU: Not required
- Bounding boxes: Yes
- Confidence scores: Yes (per word)

**Installation:**
```bash
pip install pytesseract
# + Tesseract binary (see above)
```

**Preprocessing:** Automatic (grayscale, threshold, denoise)

---

## Performance Comparison

| Feature | TrOCR | PaddleOCR 2.7.x | Tesseract |
|---------|-------|-----------------|-----------|
| **Speed** | Moderate | Fast | Very Fast |
| **Handwritten** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Printed Text** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Digits** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Multi-language** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **GPU Support** | ✅ | ✅ | ❌ |
| **Bounding Boxes** | ❌ | ✅ | ✅ |
| **Confidence** | ❌ | ✅ | ✅ |

## Use Cases

### Wagon Number Detection

**Recommended:** PaddleOCR 2.7.x or Tesseract (Digits Only)

```python
# PaddleOCR
engine = PaddleOCREngine(use_gpu=True)
result = engine.extract_text(wagon_image)

# Tesseract (digits only)
engine = TesseractEngine(digits_only=True)
result = engine.extract_text(wagon_image)
```

### Handwritten Notes

**Recommended:** TrOCR

```python
engine = TrOCREngine()
result = engine.extract_text(handwritten_image)
```

### Document Scanning

**Recommended:** PaddleOCR 2.7.x

```python
engine = PaddleOCREngine(use_gpu=True)
result = engine.extract_text(document_image)
```

## Architecture

### Modular Design

```
ocr_engines.py
├── OCREngine (Abstract Base Class)
│   ├── initialize()
│   ├── extract_text()
│   └── get_name()
├── TrOCREngine
├── PaddleOCREngine
└── TesseractEngine

ocr_comparison_app.py
├── Streamlit UI
├── Engine selection
├── Image input
├── Result display
└── Caching (@st.cache_resource)
```

### Key Features

1. **Modular Engines**
   - Each engine is a separate class
   - Unified interface
   - Easy to add new engines

2. **Caching**
   - Models loaded once
   - Cached with `@st.cache_resource`
   - No reloading on UI interactions

3. **Error Handling**
   - Graceful failures
   - Clear error messages
   - Installation instructions

4. **Performance**
   - Efficient preprocessing
   - GPU support where available
   - Minimal memory footprint

## Troubleshooting

### TrOCR Issues

**Problem:** Out of memory
```bash
# Use CPU instead of GPU
device = torch.device('cpu')
```

**Problem:** Slow processing
```bash
# Use smaller model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
```

### PaddleOCR Issues

**Problem:** Version 3.x installed
```bash
pip uninstall paddleocr
pip install paddleocr==2.7.0.3
```

**Problem:** GPU not detected
```bash
pip install paddlepaddle-gpu
```

### Tesseract Issues

**Problem:** Tesseract not found
```bash
# Windows: Add to PATH or set in code
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Problem:** Poor accuracy
```bash
# Enable preprocessing in UI
# Or adjust preprocessing parameters in code
```

## Advanced Usage

### Batch Processing

```python
from ocr_engines import create_ocr_engine

# Initialize engine
engine = create_ocr_engine('paddleocr', use_gpu=True)
engine.initialize()

# Process multiple images
for image_path in image_paths:
    image = cv2.imread(image_path)
    result = engine.extract_text(image)
    print(f"{image_path}: {result['text']}")
```

### Custom Preprocessing

```python
# Add custom preprocessing to TesseractEngine
class CustomTesseractEngine(TesseractEngine):
    def preprocess_image(self, image):
        # Your custom preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ... more processing
        return processed
```

### Integration with Video Pipeline

```python
# In your video processing pipeline
from ocr_engines import create_ocr_engine

# Initialize once
ocr_engine = create_ocr_engine('paddleocr', use_gpu=True)
ocr_engine.initialize()

# Process each frame
for frame in video_frames:
    result = ocr_engine.extract_text(frame)
    # Use result['text'], result['boxes'], etc.
```

## API Reference

### OCREngine.extract_text()

**Returns:**
```python
{
    'text': str,              # Extracted text
    'confidence': float,      # Average confidence (0-1)
    'boxes': List[List],      # Bounding boxes (if available)
    'processing_time': float, # Time in seconds
    'success': bool,          # Success status
    'details': List[Tuple],   # (text, confidence) pairs (if available)
    'error': str              # Error message (if failed)
}
```

## Best Practices

1. **Choose the Right Engine**
   - Handwritten → TrOCR
   - Printed text → PaddleOCR 2.7.x
   - Digits/Numbers → Tesseract (digits only)

2. **Use GPU When Available**
   - TrOCR: Significant speedup
   - PaddleOCR: 2-3x faster
   - Tesseract: No GPU support

3. **Preprocess Images**
   - Tesseract benefits most from preprocessing
   - PaddleOCR has built-in preprocessing
   - TrOCR is robust to image quality

4. **Cache Models**
   - Load models once
   - Reuse for multiple images
   - Use Streamlit caching

## Summary

The OCR Comparison Tool provides:
- ✅ Three powerful OCR engines
- ✅ Easy-to-use Streamlit interface
- ✅ Modular, extensible architecture
- ✅ Performance metrics and visualization
- ✅ Production-ready code

Perfect for:
- Testing different OCR engines
- Comparing accuracy and speed
- Choosing the best engine for your use case
- Integrating into existing pipelines
