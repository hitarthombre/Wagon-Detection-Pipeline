# Enhanced Video Processing Pipeline

OpenCV + YOLOv8 video processing with confidence scores, enhanced OCR, and automatic result storage.

## 🎯 Features

✅ **Manual Confidence Display** - See exact detection confidence (0.00-1.00)  
✅ **Green Bounding Boxes** - Clear visualization with 2-decimal precision  
✅ **Enhanced OCR** - 15-25% accuracy improvement with preprocessing  
✅ **Dual OCR Engines** - Choose between EasyOCR or PaddleOCR (2-3x faster)  
✅ **Output Storage** - Automatic saving to `output/` folder  
✅ **Headless Processing** - Works without GUI support  
✅ **Fast Mode** - 15-20 FPS without OCR  
✅ **Complete Mode** - Full pipeline with OCR  

## 🚀 Quick Start

### Fast Processing (Recommended)
```bash
python quick_process.py video/video_test_1.mp4
```

**Output:** Video with confidence scores + detection results  
**Speed:** ~15-20 FPS

### With OCR
```bash
# Using EasyOCR (default)
python integrated_pipeline_headless.py video/video_test_1.mp4

# Using PaddleOCR (2-3x faster)
python integrated_pipeline_headless.py video/video_test_1.mp4 yolov8n.pt 100 0.25 True True True True paddleocr
```

**Output:** Video + detections + OCR results  
**Speed:** ~2-4 FPS (EasyOCR) / ~5-7 FPS (PaddleOCR)

### Interactive Dashboard
```bash
streamlit run app.py
```

**Output:** Web-based GUI with real-time preview

### OCR Comparison Tool (NEW) 🆕
```bash
streamlit run ocr_comparison_app.py
```

**Features:**
- Compare TrOCR, PaddleOCR 2.7.x, and Tesseract
- Upload images or use webcam
- Real-time OCR with confidence scores
- Bounding box visualization
- Performance metrics

## 📦 Installation

```bash
# Basic installation
pip install opencv-python ultralytics easyocr numpy streamlit

# Optional: Install PaddleOCR (2-3x faster OCR)
pip install paddleocr
```

## 📊 Output

All results saved to `output/` folder:

- `{video_name}_processed.mp4` - Video with confidence scores
- `{video_name}_detections.txt` - Frame-by-frame detections
- `{video_name}_ocr_results.txt` - OCR text extraction

## 🎬 Example Output

### Video
Green bounding boxes with labels like:
```
wagon: 0.87
person: 0.63
car: 0.30
```

### Detection Results
```
Frame 1 - Detections: 2
  1. person: 0.63 at [1607, 718, 1652, 844]
  2. car: 0.30 at [633, 720, 698, 774]

SUMMARY
Total frames: 706
Total detections: 2,321
Average: 3.29 per frame
```

## ⚙️ Configuration

```bash
# Adjust confidence threshold
python quick_process.py video.mp4 yolov8n.pt 100 0.30

# More detections (lower threshold)
python quick_process.py video.mp4 yolov8n.pt 100 0.15

# Fewer, confident detections (higher threshold)
python quick_process.py video.mp4 yolov8n.pt 100 0.50
```

## 📚 Documentation

- **USAGE_GUIDE.md** - Detailed usage instructions
- **PADDLEOCR_GUIDE.md** - PaddleOCR integration guide
- **README_ENHANCEMENTS.md** - Technical implementation
- **QUICKSTART_ENHANCED.md** - Quick examples
- **FEATURE_COMPARISON.md** - Before/after comparison
- **IMPLEMENTATION_COMPLETE.md** - Project summary

## 🧪 Testing

```bash
python test_enhancements.py
```

Expected: All 5 tests pass ✅

## 📈 Performance

| Mode | Speed | OCR | Output |
|------|-------|-----|--------|
| Fast | 15-20 FPS | ❌ | Video + Detections |
| Complete (EasyOCR) | 2-4 FPS | ✅ | Video + Detections + OCR |
| Complete (PaddleOCR) | 5-7 FPS | ✅ | Video + Detections + OCR |
| Dashboard | 3-5 FPS | ✅ | Web Interface |

## 🎯 Use Cases

**Demo/Presentation**
```bash
python quick_process.py video/video_test_1.mp4
start output\video_test_1_processed.mp4
```

**Complete Analysis**
```bash
# With EasyOCR
python integrated_pipeline_headless.py video/video_test_1.mp4
type output\video_test_1_detections.txt
type output\video_test_1_ocr_results.txt

# With PaddleOCR (faster)
python integrated_pipeline_headless.py video/video_test_1.mp4 yolov8n.pt 100 0.25 True True True True paddleocr
```

**Batch Processing**
```bash
for %f in (video\*.mp4) do python quick_process.py %f
```

## 🐛 Troubleshooting

**GUI Error:** Use headless scripts (`quick_process.py` or `integrated_pipeline_headless.py`)  
**Slow Processing:** Use `quick_process.py` (no OCR) or disable OCR  
**Missing Detections:** Lower confidence threshold (0.15-0.20)  
**Too Many False Positives:** Raise confidence threshold (0.40-0.50)  

## 📁 Project Structure

```
├── quick_process.py                 # Fast processing (recommended)
├── integrated_pipeline_headless.py  # Complete pipeline with OCR
├── app.py                           # Streamlit dashboard
├── step4_object_detection.py        # YOLO detection module
├── step5_ocr_extraction.py          # OCR extraction module
├── test_enhancements.py             # Test suite
├── output/                          # Results folder (auto-created)
│   ├── *_processed.mp4
│   ├── *_detections.txt
│   └── *_ocr_results.txt
└── video/                           # Input videos
    ├── video_test_1.mp4
    ├── video_test_2.mp4
    └── video_test_3.mp4
```

## ✅ Status

**Version:** 2.0 (Enhanced Edition)  
**Status:** Production Ready  
**Tests:** 5/5 Passing  
**Features:** All Implemented  

## 🎓 Learn More

See **USAGE_GUIDE.md** for comprehensive instructions and examples.

---

**Quick Commands:**
```bash
# Fast demo
python quick_process.py video/video_test_1.mp4

# Complete analysis (EasyOCR)
python integrated_pipeline_headless.py video/video_test_1.mp4

# Complete analysis (PaddleOCR - faster)
python integrated_pipeline_headless.py video/video_test_1.mp4 yolov8n.pt 100 0.25 True True True True paddleocr

# Interactive
streamlit run app.py

# Test
python test_enhancements.py
```
