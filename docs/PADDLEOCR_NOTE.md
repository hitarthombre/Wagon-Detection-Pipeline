# PaddleOCR Compatibility Note

## Current Status

PaddleOCR 3.0+ has been integrated into the pipeline, but there are known compatibility issues with certain system configurations.

## Known Issues

### Issue 1: API Changes
PaddleOCR 3.0 introduced breaking API changes:
- `use_gpu` parameter removed → use `device='gpu'` or `device='cpu'`
- `use_angle_cls` deprecated → use `use_textline_orientation`
- `show_log` parameter removed
- `ocr()` method deprecated → use `predict()`

### Issue 2: Runtime Errors
Some systems may encounter:
```
NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
```

This appears to be related to oneDNN/MKL-DNN compatibility issues on certain Windows configurations.

## Recommendation

**For most users: Use EasyOCR (default)**

EasyOCR is:
- ✅ More stable across different systems
- ✅ Well-tested and reliable
- ✅ Easier to install and configure
- ✅ Good accuracy for most use cases

## If You Want to Try PaddleOCR

### Installation
```bash
pip install paddleocr
```

### Usage
```bash
# Test if it works on your system
python compare_ocr_engines.py video/video_test_1.mp4 3

# If successful, use in pipeline
python step5_ocr_extraction.py video/video_test_1.mp4 100 True True paddleocr
```

### Troubleshooting

**If you get API errors:**
- The code automatically falls back to EasyOCR
- No action needed

**If you get runtime errors:**
- PaddleOCR may not be compatible with your system
- Use EasyOCR instead (default)
- The pipeline will work perfectly with EasyOCR

## Performance Comparison

When PaddleOCR works:
- **Speed:** 2-3x faster than EasyOCR
- **Accuracy:** ~5% better on average

However, stability and compatibility are more important than raw speed for most users.

## Current Recommendation

**Stick with EasyOCR** unless you:
1. Have tested PaddleOCR successfully on your system
2. Need the extra speed for batch processing
3. Are comfortable troubleshooting compatibility issues

## Future Updates

We're monitoring PaddleOCR development and will update the integration as the library matures and compatibility improves.

## Summary

- **Default:** EasyOCR (recommended for stability)
- **Optional:** PaddleOCR (if it works on your system)
- **Fallback:** Automatic fallback to EasyOCR if PaddleOCR fails

The pipeline is designed to work reliably with EasyOCR, with PaddleOCR as an optional performance enhancement for compatible systems.
