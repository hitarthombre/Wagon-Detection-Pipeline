# Usage Guide - Enhanced Video Processing Pipeline

## ⚠️ Important: GUI vs Headless Mode

Your environment doesn't support OpenCV's GUI (`cv2.imshow`). This is common in:
- Headless servers
- Docker containers
- Some Windows configurations
- Remote desktop sessions

**Solution**: Use the headless versions that save output videos instead of displaying them.

## 🚀 Quick Start (Recommended)

### Fast Processing (No OCR)
For quick results with confidence scores:

```bash
python quick_process.py video/video_test_1.mp4
```

**Output:**
- `output/video_test_1_processed.mp4` - Video with confidence scores
- `output/video_test_1_detections.txt` - Detection results

**Processing Speed:** ~15-20 FPS (fast!)

---

### Full Pipeline (With OCR)
For complete processing including text extraction:

```bash
python integrated_pipeline_headless.py video/video_test_1.mp4 yolov8n.pt 100 0.25 True True True False
```

**Parameters:**
1. `video/video_test_1.mp4` - Video path
2. `yolov8n.pt` - YOLO model
3. `100` - Blur threshold
4. `0.25` - Confidence threshold
5. `True` - Enable OCR
6. `True` - Use GPU for OCR
7. `True` - Save text results
8. `False` - Don't save video (faster)

**Output:**
- `output/video_test_1_detections.txt` - Detection results
- `output/video_test_1_ocr_results.txt` - OCR results
- `output/video_test_1_processed.mp4` - Video (if enabled)

**Processing Speed:** ~2-4 FPS (slower due to OCR)

---

## 📊 Available Scripts

### 1. `quick_process.py` ⚡ (RECOMMENDED)
**Best for:** Fast demos, testing, confidence score visualization

```bash
# Basic usage
python quick_process.py video/video_test_1.mp4

# Custom confidence threshold
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.30

# Lower threshold for more detections
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.15
```

**Features:**
✅ Manual confidence display (2 decimals)
✅ Green bounding boxes
✅ Fast processing (~15-20 FPS)
✅ Saves output video
✅ Saves detection results
❌ No OCR (for speed)
❌ No GUI display

---

### 2. `integrated_pipeline_headless.py` 🔬
**Best for:** Complete analysis with OCR

```bash
# Full processing with OCR
python integrated_pipeline_headless.py video/video_test_1.mp4

# Without OCR (faster)
python integrated_pipeline_headless.py video/video_test_1.mp4 yolov8n.pt 100 0.25 False

# Save results only (no video)
python integrated_pipeline_headless.py video/video_test_1.mp4 yolov8n.pt 100 0.25 True True True False
```

**Features:**
✅ Manual confidence display
✅ Enhanced OCR preprocessing
✅ Dual OCR processing
✅ Saves output video (optional)
✅ Saves detection + OCR results
✅ Progress bar
❌ Slower with OCR (~2-4 FPS)
❌ No GUI display

---

### 3. `app.py` 🖥️ (Streamlit Dashboard)
**Best for:** Interactive web interface

```bash
streamlit run app.py
```

**Features:**
✅ Web-based GUI
✅ Real-time preview
✅ Interactive controls
✅ Multiple pipeline steps
✅ OCR data display
✅ Works in browser
❌ Requires Streamlit

---

### 4. `step4_object_detection.py` 🎯
**Best for:** Testing YOLO detection only

```bash
python step4_object_detection.py video/video_test_1.mp4
```

**Note:** Requires GUI support. Use `quick_process.py` instead if you get display errors.

---

### 5. `step5_ocr_extraction.py` 📝
**Best for:** Testing OCR only

```bash
python step5_ocr_extraction.py video/video_test_1.mp4
```

**Note:** Requires GUI support. Use `integrated_pipeline_headless.py` instead if you get display errors.

---

## 🎬 Viewing Results

### Output Video
The processed video includes:
- Green bounding boxes around detections
- Confidence scores (e.g., "wagon: 0.87")
- Frame information overlay
- Blur status
- Detection count

**View with any video player:**
```bash
# Windows
start output\video_test_1_processed.mp4

# Or use VLC, Windows Media Player, etc.
```

### Detection Results
Text file with frame-by-frame detections:

```bash
type output\video_test_1_detections.txt
```

**Format:**
```
Frame 1 - Detections: 3
  1. wagon: 0.87 at [120, 45, 340, 280]
  2. wagon: 0.92 at [450, 60, 670, 295]
  3. train: 0.85 at [200, 100, 500, 400]

Frame 2 - Detections: 2
  ...

SUMMARY
Total frames: 706
Total detections: 2321
Average detections per frame: 3.29
```

### OCR Results
Text file with extracted text:

```bash
type output\video_test_1_ocr_results.txt
```

**Format:**
```
Frame 1 - Text detected: 2
  1. 'WAGON' (confidence: 0.92)
  2. '12345' (confidence: 0.88)

Frame 5 - Text detected: 1
  1. 'FREIGHT' (confidence: 0.85)

SUMMARY
Total text instances: 420
Average text per frame: 2.80
```

---

## ⚙️ Configuration

### Confidence Threshold
Controls detection sensitivity:

```bash
# More detections (may include false positives)
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.15

# Balanced (default)
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.25

# Fewer, more confident detections
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.50
```

### Blur Threshold
Controls enhancement sensitivity:

```bash
# More sensitive (enhance more frames)
python quick_process.py video/video_test_1.mp4 yolov8n.pt 80 0.25

# Default
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.25

# Less sensitive (enhance fewer frames)
python quick_process.py video/video_test_1.mp4 yolov8n.pt 150 0.25
```

---

## 📦 Batch Processing

### Process All Videos
```bash
# Windows
for %f in (video\*.mp4) do python quick_process.py %f

# PowerShell
Get-ChildItem video\*.mp4 | ForEach-Object { python quick_process.py $_.FullName }
```

### Process with Different Thresholds
```bash
# Test multiple confidence thresholds
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.15
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.25
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.35
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.50
```

---

## 🐛 Troubleshooting

### Error: "The function is not implemented"
**Problem:** OpenCV GUI not available

**Solution:** Use headless scripts:
- `quick_process.py` (fast, no OCR)
- `integrated_pipeline_headless.py` (with OCR)
- `app.py` (Streamlit web interface)

### Error: "Cannot open video"
**Problem:** Video file not found

**Solution:**
```bash
# Check video exists
dir video\*.mp4

# Use full path
python quick_process.py "D:\path\to\video.mp4"
```

### Slow Processing
**Problem:** OCR is very slow

**Solutions:**
1. Use `quick_process.py` (no OCR)
2. Disable OCR: `python integrated_pipeline_headless.py video.mp4 yolov8n.pt 100 0.25 False`
3. Don't save video: `python integrated_pipeline_headless.py video.mp4 yolov8n.pt 100 0.25 True True True False`
4. Ensure GPU is available for YOLO and OCR

### Too Many False Detections
**Problem:** Low confidence detections

**Solution:** Increase confidence threshold:
```bash
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.40
```

### Missing Detections
**Problem:** High confidence threshold

**Solution:** Lower confidence threshold:
```bash
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.15
```

---

## 📈 Performance Comparison

| Script | Speed (FPS) | OCR | GUI | Output Video | Best For |
|--------|-------------|-----|-----|--------------|----------|
| `quick_process.py` | 15-20 | ❌ | ❌ | ✅ | Fast demos |
| `integrated_pipeline_headless.py` | 2-4 | ✅ | ❌ | ✅ | Complete analysis |
| `app.py` | 3-5 | ✅ | ✅ | ❌ | Interactive use |
| `step4_object_detection.py` | 15-20 | ❌ | ✅ | ❌ | Testing (needs GUI) |
| `step5_ocr_extraction.py` | 2-4 | ✅ | ✅ | ❌ | Testing (needs GUI) |

---

## 🎯 Use Case Examples

### Demo/Presentation
```bash
# Quick processing for demo
python quick_process.py video/video_test_1.mp4

# View output video
start output\video_test_1_processed.mp4

# Show confidence scores to audience
```

### Development/Testing
```bash
# Test different thresholds quickly
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.20
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.30
python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.40

# Compare results
type output\video_test_1_detections.txt
```

### Complete Analysis
```bash
# Full pipeline with OCR
python integrated_pipeline_headless.py video/video_test_1.mp4

# Analyze results
type output\video_test_1_detections.txt
type output\video_test_1_ocr_results.txt
```

### Batch Processing
```bash
# Process all videos quickly
for %f in (video\*.mp4) do python quick_process.py %f

# Check all results
dir output\*.txt
```

---

## 💡 Tips

1. **Start with `quick_process.py`** - It's fast and shows confidence scores
2. **Use OCR only when needed** - It's much slower
3. **Adjust confidence threshold** - 0.25 is a good starting point
4. **Check output folder** - All results are saved there
5. **Use Streamlit for interactive work** - `streamlit run app.py`
6. **View videos with any player** - VLC, Windows Media Player, etc.

---

## 📚 Documentation

- **README_ENHANCEMENTS.md** - Technical details
- **QUICKSTART_ENHANCED.md** - Quick start guide
- **FEATURE_COMPARISON.md** - Before/after comparison
- **CHANGES_SUMMARY.md** - Complete change log
- **USAGE_GUIDE.md** - This file

---

## ✅ Summary

**For fast results with confidence scores:**
```bash
python quick_process.py video/video_test_1.mp4
```

**For complete analysis with OCR:**
```bash
python integrated_pipeline_headless.py video/video_test_1.mp4
```

**For interactive web interface:**
```bash
streamlit run app.py
```

All scripts save results to the `output/` folder!
