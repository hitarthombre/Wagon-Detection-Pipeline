# YOLOv8 Training Guide

## Prerequisites

1. Install required packages:
```bash
pip install ultralytics torch torchvision
```

2. Download your dataset from Roboflow:
   - Export in YOLOv8 format
   - Extract the dataset
   - Ensure `data.yaml` is in the project root or note its path

## Dataset Structure

Your Roboflow dataset should have this structure:
```
project/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Training

### Basic Training (Default Settings)
```bash
python train_yolov8.py
```

This uses:
- Dataset: `data.yaml`
- Epochs: 50
- Image size: 640
- Batch size: 16
- Base model: yolov8n.pt

### Custom Training
```bash
python train_yolov8.py <data_yaml_path> <epochs> <img_size> <batch_size>
```

Example:
```bash
python train_yolov8.py dataset/data.yaml 100 640 8
```

## Training Parameters

- **Epochs**: 50 (default) - Increase for better accuracy, decrease for faster training
- **Image Size**: 640 (default) - Larger = better accuracy but slower
- **Batch Size**: 16 (default) - Adjust based on GPU memory (RTX 3050 can handle 16-32)
- **Base Model**: yolov8n.pt (nano) - Fast and lightweight

Other model options:
- `yolov8s.pt` - Small (more accurate, slower)
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra large

## GPU Acceleration

The script automatically uses your RTX 3050 GPU if available. You'll see:
```
Device: cuda
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

## Output

After training, you'll find:
- **Best model**: `runs/detect/train/weights/best.pt`
- **Last model**: `runs/detect/train/weights/last.pt`
- **Training plots**: `runs/detect/train/`
- **Metrics**: Confusion matrix, F1 curve, PR curve, etc.

## Using Trained Model

### In standalone script:
```bash
python step4_object_detection.py video.mp4 runs/detect/train/weights/best.pt
```

### In Streamlit app:
Update the model path in `app.py` to use your trained model.

## Monitoring Training

Training progress shows:
- Epoch number
- Loss values (box, cls, dfl)
- Metrics (precision, recall, mAP50, mAP50-95)
- Learning rate
- GPU memory usage

## Tips for Better Results

1. **More data**: 500+ images per class recommended
2. **Data augmentation**: Enabled by default in YOLOv8
3. **Longer training**: Try 100-200 epochs for complex datasets
4. **Larger model**: Use yolov8s.pt or yolov8m.pt for better accuracy
5. **Adjust batch size**: Increase if you have GPU memory available
6. **Early stopping**: Enabled with patience=10 (stops if no improvement)

## Troubleshooting

### Out of Memory Error
Reduce batch size:
```bash
python train_yolov8.py data.yaml 50 640 8
```

### Slow Training
- Use smaller image size (416 or 512)
- Use yolov8n.pt instead of larger models
- Reduce batch size

### Poor Results
- Check dataset quality and labels
- Increase epochs
- Use larger model (yolov8s.pt or yolov8m.pt)
- Ensure balanced dataset (similar number of images per class)
