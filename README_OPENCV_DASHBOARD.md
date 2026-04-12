# OpenCV Dashboard - AI Video Processing Pipeline

## Overview
Real-time OpenCV dashboard that visualizes all stages of the AI video processing pipeline in a single window with a 2x2 grid layout.

## Features

### Visual Layout
```
┌─────────────────────┬─────────────────────┐
│  Step 1: Original   │  Step 2: Blur Det.  │
│                     │                     │
├─────────────────────┼─────────────────────┤
│  Step 3: Enhanced   │  Step 4: OCR        │
│                     │                     │
└─────────────────────┴─────────────────────┘
```

### Pipeline Stages
1. **Original Frame** - Raw video input
2. **Blur Detection** - Laplacian variance with score and status
3. **Enhancement** - Conditional sharpening for blurry frames
4. **OCR Extraction** - Text detection with bounding boxes

### Real-time Statistics
- Frame counter and progress
- Processing FPS
- Clear vs Blurry frame counts
- Enhancement statistics
- Total text detected

## Usage

### Basic Usage
```bash
python opencv_dashboard.py
```
Uses default video: `video/video_test_1.mp4`

### With Custom Video
```bash
python opencv_dashboard.py path/to/video.mp4
```

### With Parameters
```bash
python opencv_dashboard.py video.mp4 100 0.7
```
- Argument 1: Video path
- Argument 2: Blur threshold (default: 100.0)
- Argument 3: OCR confidence threshold (default: 0.7)

### Disable OCR (Faster Processing)
```bash
python opencv_dashboard.py video.mp4 100 0.7 false
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit/Stop processing |
| `p` | Pause/Resume |

## Display Components

### Step 1: Original Frame
- Shows raw video input
- No processing applied

### Step 2: Blur Detection
- Displays blur score (Laplacian variance)
- Shows status: "Clear" (green) or "Blurry" (red)
- Color-coded for quick identification

### Step 3: Enhancement
- Shows enhanced or original frame
- Label indicates: "ENHANCED" (orange) or "ORIGINAL" (green)
- Only blurry frames are enhanced for efficiency

### Step 4: OCR Extraction
- Green bounding boxes around detected text
- Text labels with confidence scores
- Counter shows number of text instances found

### Statistics Overlay
Located in top-left corner:
- **Frame**: Current/Total frames
- **FPS**: Processing speed
- **Clear**: Number of clear frames
- **Blurry**: Number of blurry frames
- **Enhanced**: Frames that were enhanced
- **Text**: Total text instances detected

## Output Summary

After processing completes, displays:
- Total frames processed
- Processing time and average FPS
- Blur detection statistics (percentages)
- Enhancement efficiency
- OCR statistics (if enabled)

## Performance Tips

1. **Disable OCR for faster processing**:
   ```bash
   python opencv_dashboard.py video.mp4 100 0.7 false
   ```

2. **Adjust display size** (edit in code):
   ```python
   display_size=(640, 480)  # Smaller = faster
   ```

3. **Use GPU acceleration**:
   - EasyOCR automatically uses GPU if available
   - Ensure CUDA is properly configured

## Technical Details

### Grid Layout
- 2x2 grid showing all 4 pipeline stages
- Each frame is 640x480 pixels (configurable)
- Total window size: 1280x960 pixels
- Labels added to each section

### Processing Flow
```
Video Input
    ↓
Original Frame (Step 1)
    ↓
Blur Detection (Step 2) → Score & Status
    ↓
Enhancement (Step 3) → Conditional Processing
    ↓
OCR Extraction (Step 4) → Text Detection
    ↓
Grid Display with Stats
```

### Color Coding
- **Green**: Clear frames, original (not enhanced)
- **Red**: Blurry frames
- **Orange**: Enhanced frames
- **Green boxes**: Detected text

## Requirements
- opencv-contrib-python >= 4.8.0
- numpy >= 1.24.0
- easyocr >= 1.7.0
- torch (for GPU support)

## Comparison with Streamlit Dashboard

| Feature | OpenCV Dashboard | Streamlit Dashboard |
|---------|-----------------|---------------------|
| Display | Single window, grid | Multi-column web UI |
| Performance | Faster | Slower (web overhead) |
| Interactivity | Keyboard only | Full UI controls |
| Deployment | Local only | Web accessible |
| Best for | Development/Testing | Production/Demo |

## Troubleshooting

### Window not displaying
- Check if OpenCV is built with GUI support
- On Linux: Install `libgtk2.0-dev`
- On Windows: Reinstall opencv-python

### Slow processing
- Disable OCR: Add `false` as 4th argument
- Reduce display size in code
- Use smaller video resolution

### OCR not working
- Ensure EasyOCR is installed: `pip install easyocr`
- Check GPU availability
- Verify CUDA installation for GPU mode

## Examples

### Quick test with default settings
```bash
python opencv_dashboard.py
```

### High-quality blur detection
```bash
python opencv_dashboard.py video.mp4 150
```

### Fast processing without OCR
```bash
python opencv_dashboard.py video.mp4 100 0.7 false
```

### Strict OCR confidence
```bash
python opencv_dashboard.py video.mp4 100 0.9
```

## Future Enhancements
- [ ] Add object detection (YOLOv8) as 5th stage
- [ ] Export processed video
- [ ] Save OCR results to CSV
- [ ] Add frame-by-frame navigation
- [ ] Support for multiple video formats
- [ ] Real-time parameter adjustment
