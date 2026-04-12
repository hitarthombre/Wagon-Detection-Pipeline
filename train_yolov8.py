"""
YOLOv8 Training Script for Custom Object Detection
Train a custom YOLOv8 model using Roboflow dataset
"""

import os
from ultralytics import YOLO
import torch

def train_yolov8_model(
    data_yaml_path='data.yaml',
    base_model='yolov8n.pt',
    epochs=50,
    img_size=640,
    batch_size=16,
    project_name='runs/detect',
    name='train'
):
    """
    Train YOLOv8 model on custom dataset.
    
    Args:
        data_yaml_path: Path to data.yaml file from Roboflow
        base_model: Pre-trained model to start from
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size for training
        project_name: Project directory name
        name: Experiment name
    
    Returns:
        Path to best.pt model
    """
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🚀 YOLOv8 Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Base Model: {base_model}")
    print(f"Dataset: {data_yaml_path}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {img_size}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: {data_yaml_path} not found!")
        print(f"Please ensure your Roboflow dataset is downloaded and data.yaml is in the correct location.")
        return None
    
    # Load pre-trained YOLOv8 model
    print(f"📦 Loading pre-trained model: {base_model}")
    model = YOLO(base_model)
    print(f"✅ Model loaded successfully\n")
    
    # Start training
    print(f"🎯 Starting training...")
    print(f"{'='*60}\n")
    
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=project_name,
            name=name,
            patience=10,  # Early stopping patience
            save=True,
            plots=True,
            verbose=True,
            val=True,
            cache=False,  # Set to True if you have enough RAM
            workers=0,  # Set to 0 for Windows to avoid multiprocessing issues
            optimizer='auto',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            amp=True  # Automatic Mixed Precision
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Training completed successfully!")
        print(f"{'='*60}\n")
        
        # Get path to best model
        best_model_path = os.path.join(project_name, name, 'weights', 'best.pt')
        last_model_path = os.path.join(project_name, name, 'weights', 'last.pt')
        
        # Check if files exist
        if not os.path.exists(best_model_path):
            print(f"\n⚠️ Warning: best.pt not found at expected location")
            print(f"   Searching for model files...")
            # Try to find the actual location
            for root, dirs, files in os.walk(project_name):
                if 'best.pt' in files:
                    best_model_path = os.path.join(root, 'best.pt')
                    print(f"   Found at: {best_model_path}")
                    break
        
        if not os.path.exists(best_model_path):
            print(f"\n❌ Error: Could not find trained model")
            return None
        
        print(f"📊 Training Results:")
        print(f"   Best Model: {os.path.abspath(best_model_path)}")
        if os.path.exists(last_model_path):
            print(f"   Last Model: {os.path.abspath(last_model_path)}")
        print(f"   Results: {os.path.abspath(os.path.join(project_name, name))}")
        
        # Print metrics if available
        if hasattr(results, 'results_dict'):
            print(f"\n📈 Final Metrics:")
            metrics = results.results_dict
            if 'metrics/mAP50(B)' in metrics:
                print(f"   mAP50: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"   mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
        
        print(f"\n{'='*60}")
        print(f"🎉 Model training complete!")
        print(f"{'='*60}\n")
        
        return best_model_path
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        return None

def validate_model(model_path, data_yaml_path='model/data.yaml', img_size=640):
    """
    Validate trained model on test set.
    
    Args:
        model_path: Path to trained model
        data_yaml_path: Path to data.yaml
        img_size: Image size for validation
    """
    print(f"\n{'='*60}")
    print(f"🔍 Validating Model")
    print(f"{'='*60}\n")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return None
    
    try:
        model = YOLO(model_path)
        results = model.val(data=data_yaml_path, imgsz=img_size)
        
        print(f"\n✅ Validation complete!")
        return results
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        return None

if __name__ == "__main__":
    import sys
    
    # Default parameters
    data_yaml = 'model/data.yaml'
    base_model = 'yolov8n.pt'
    epochs = 50
    img_size = 640
    batch_size = 16
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        data_yaml = sys.argv[1]
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    if len(sys.argv) > 3:
        img_size = int(sys.argv[3])
    if len(sys.argv) > 4:
        batch_size = int(sys.argv[4])
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         YOLOv8 Custom Model Training Script             ║
║              Wagon Detection Pipeline                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Train the model
    best_model = train_yolov8_model(
        data_yaml_path=data_yaml,
        base_model=base_model,
        epochs=epochs,
        img_size=img_size,
        batch_size=batch_size
    )
    
    if best_model:
        print(f"\n💾 Best model saved at: {best_model}")
        print(f"\n📝 Usage in pipeline:")
        print(f"   python step4_object_detection.py video.mp4 {best_model}")
        
        # Optional: Run validation
        validate = input("\n🔍 Run validation on test set? (y/n): ").lower()
        if validate == 'y':
            validate_model(best_model, data_yaml, img_size)
    else:
        print(f"\n❌ Training failed. Please check the error messages above.")
        sys.exit(1)
