# Quick Start Guide - Enhanced Pipeline

## Installation
Ensure you have all dependencies installed:
```bash
pip install opencv-python ultralytics easyocr numpy
```

## Quick Test

### 1. Run Integrated Pipeline (Recommended)
Process video with all enhancements enabled:
```bash
python integrated_pipeline.py video/video_test_1.mp4
```

**What you'll see:**
- Real-time video processing with all pipeline steps
- Green bounding boxes with confidence scores (e.g., "wagon: 0.87")
- OCR text detection with enhanced preprocessing
- Frame statistics overlay
- Pause indicator when paused

**Controls:**
- Press `p` to pause
- Press `r` to resume
- Press `q` to quit

**Output:**
- `output/video_test_1_detections.txt` - All YOLO detections
- `output/video_test_1_ocr_results.txt` - All OCR text extractions

### 2. Run Object Detection Only
Test YOLO detection with manual confidence display:
```bash
python step4_object_detection.py video/video_test_1.mp4
```

### 3. Run OCR Only
Test enhanced OCR with preprocessing:
```bash
python step5_ocr_extraction.py video/video_test_1.mp4
```

## Key Features Demo

### Confidence Score Display
Watch for green boxes around detected objects with labels like:
```
wagon: 0.87
train: 0.92
```

### Play/Pause Controls
1. Start video processing
2. Press `p` when you see an interesting frame
3. Examine the detections and confidence scores
4. Press `r` to continue
5. Press `q` to exit

### Enhanced OCR
The system now:
- Preprocesses frames for better text clarity
- Runs OCR on both original and preprocessed versions
- Keeps the best results
- Displays text with confidence scores

### Output Files
Check the `output/` folder after processing:
```bash
# View detection results
type output\video_test_1_detections.txt

# View OCR results
type output\video_test_1_ocr_results.txt
```

## Customization

### Adjust Detection Confidence
Lower threshold = more detections (may include false positives):
```bash
python integrated_pipeline.py video/video_test_1.mp4 yolov8n.pt 100 0.15
```

Higher threshold = fewer, more confident detections:
```bash
python integrated_pipeline.py video/video_test_1.mp4 yolov8n.pt 100 0.50
```

### Disable OCR (Faster Processing)
```bash
python integrated_pipeline.py video/video_test_1.mp4 yolov8n.pt 100 0.25 False
```

### Disable Output Saving
```bash
python integrated_pipeline.py video/video_test_1.mp4 yolov8n.pt 100 0.25 True True False
```

## Troubleshooting

### Issue: OCR is slow
**Solution:** Disable OCR or use GPU acceleration:
```bash
python integrated_pipeline.py video/video_test_1.mp4 yolov8n.pt 100 0.25 True True True
```

### Issue: Too many false detections
**Solution:** Increase confidence threshold:
```bash
python integrated_pipeline.py video/video_test_1.mp4 yolov8n.pt 100 0.40
```

### Issue: Missing detections
**Solution:** Lower confidence threshold:
```bash
python integrated_pipeline.py video/video_test_1.mp4 yolov8n.pt 100 0.20
```

### Issue: Blurry frames not enhanced
**Solution:** Adjust blur threshold (lower = more sensitive):
```bash
python integrated_pipeline.py video/video_test_1.mp4 yolov8n.pt 80 0.25
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for YOLO and OCR
2. **Adjust Resolution**: Process smaller frames for faster results
3. **Skip OCR**: Disable if you only need object detection
4. **Optimize Threshold**: Higher confidence = faster processing

## Example Workflow

### For Demo/Presentation:
```bash
# Run with all features, pause on interesting frames
python integrated_pipeline.py video/video_test_1.mp4
# Press 'p' to pause and show confidence scores
# Press 'r' to continue
```

### For Batch Processing:
```bash
# Process all videos without OCR for speed
for %f in (video\*.mp4) do python integrated_pipeline.py %f yolov8n.pt 100 0.25 False True True
```

### For Evaluation:
```bash
# Process with all features and save results
python integrated_pipeline.py video/video_test_1.mp4 yolov8n.pt 100 0.25 True True True
# Analyze output files in output/ folder
```

## What's New

✅ **Manual Confidence Display**: See exact detection confidence (0.00-1.00)
✅ **Play/Pause Controls**: Interactive frame inspection
✅ **Enhanced OCR**: Better text recognition with preprocessing
✅ **Output Folder**: Automatic result saving to `output/`
✅ **Improved Accuracy**: Dual OCR processing for better results
✅ **Better Visualization**: Green boxes and labels for clarity

## Next Steps

1. Test with your own videos
2. Adjust parameters for optimal results
3. Review output files for analysis
4. Use pause feature for frame-by-frame inspection
5. Compare confidence scores across different videos
