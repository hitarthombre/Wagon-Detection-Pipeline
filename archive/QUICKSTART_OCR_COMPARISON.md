# Quick Start: OCR Comparison Tool

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies

```bash
pip install -r requirements_ocr_comparison.txt
```

**Note:** For Tesseract, also install the binary:
- Windows: [Download here](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`

### 2. Test Installation

```bash
python test_ocr_engines.py
```

Expected output:
```
✅ PASS - TrOCR
✅ PASS - PaddleOCR
✅ PASS - Tesseract
✅ PASS - Tesseract (Digits)

🎉 All OCR engines are working correctly!
```

### 3. Launch App

```bash
streamlit run ocr_comparison_app.py
```

Opens at: `http://localhost:8501`

## 📸 Quick Demo

1. **Select Engine** (sidebar)
   - Choose: TrOCR, PaddleOCR 2.7.x, or Tesseract

2. **Upload Image**
   - Click "Browse files"
   - Select image with text

3. **Run OCR**
   - Click "🚀 Run OCR"
   - View extracted text
   - See confidence scores
   - Visualize bounding boxes

## 🎯 Use Cases

### Wagon Number Detection
```
Engine: PaddleOCR 2.7.x or Tesseract (Digits Only)
Speed: Fast (0.5-2s)
Accuracy: Excellent for digits
```

### Handwritten Notes
```
Engine: TrOCR
Speed: Moderate (2-5s)
Accuracy: Best for handwriting
```

### Document Scanning
```
Engine: PaddleOCR 2.7.x
Speed: Fast (0.5-2s)
Accuracy: Excellent for printed text
```

## 🔧 Configuration

### PaddleOCR GPU
```bash
pip install paddlepaddle-gpu
```
Then enable "Use GPU" in sidebar

### Tesseract Digits Only
Enable "Digits Only" checkbox for number recognition

### TrOCR Model
Default: `microsoft/trocr-base-handwritten`
For printed text: Change to `microsoft/trocr-base-printed` in code

## 📊 Performance

| Engine | Speed | Best For |
|--------|-------|----------|
| TrOCR | 2-5s | Handwritten |
| PaddleOCR | 0.5-2s | Printed text |
| Tesseract | 0.1-0.5s | Digits/Simple |

## 🐛 Troubleshooting

### TrOCR: Out of Memory
```python
# Use CPU instead
device = torch.device('cpu')
```

### PaddleOCR: Wrong Version
```bash
pip uninstall paddleocr
pip install paddleocr==2.7.0.3
```

### Tesseract: Not Found
```bash
# Windows: Add to PATH or set in code
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## 📚 Documentation

- **Full Guide:** `docs/OCR_COMPARISON_GUIDE.md`
- **API Reference:** See `ocr_engines.py` docstrings
- **Integration:** Examples in guide

## ✅ Summary

You now have:
- ✅ 3 OCR engines (TrOCR, PaddleOCR 2.7.x, Tesseract)
- ✅ Streamlit comparison interface
- ✅ Modular, extensible architecture
- ✅ Performance metrics and visualization
- ✅ Production-ready code

**Next:** Try with your own images and compare results!
