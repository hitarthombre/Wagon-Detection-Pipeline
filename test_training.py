"""
Quick test to verify YOLOv8 training setup
"""
import os
import torch

def main():
    from ultralytics import YOLO
    
    print("="*60)
    print("Testing YOLOv8 Training Setup")
    print("="*60)

    # Check GPU
    print(f"\n1. GPU Check:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")

    # Check data.yaml
    print(f"\n2. Dataset Check:")
    data_yaml = "model/data.yaml"
    if os.path.exists(data_yaml):
        print(f"   ✅ data.yaml found at: {data_yaml}")
        with open(data_yaml, 'r') as f:
            print(f"   Content preview:")
            for i, line in enumerate(f):
                if i < 5:
                    print(f"      {line.rstrip()}")
    else:
        print(f"   ❌ data.yaml not found")

    # Check model
    print(f"\n3. Model Check:")
    try:
        model = YOLO('yolov8n.pt')
        print(f"   ✅ YOLOv8n model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")

    # Test training for 1 epoch
    print(f"\n4. Quick Training Test (1 epoch):")
    try:
        model = YOLO('yolov8n.pt')
        results = model.train(
            data=data_yaml,
            epochs=1,
            imgsz=640,
            batch=4,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            project='test_runs',
            name='test',
            verbose=True,
            workers=0  # Fix for Windows multiprocessing
        )
        print(f"   ✅ Training test successful!")
        
        # Check if model was saved (check both locations)
        test_model_path = os.path.join('test_runs', 'test', 'weights', 'last.pt')
        if os.path.exists(test_model_path):
            print(f"   ✅ Model saved at: {test_model_path}")
        else:
            # Check user directory
            import glob
            user_runs = os.path.expanduser('~/runs/detect/test_runs/*/weights/best.pt')
            found_models = glob.glob(user_runs)
            if found_models:
                print(f"   ✅ Model saved at: {found_models[0]}")
            else:
                print(f"   ⚠️ Model not found at expected location")
            
    except Exception as e:
        print(f"   ❌ Training test failed: {e}")

    print(f"\n{'='*60}")
    print("Test complete!")
    print("="*60)

if __name__ == '__main__':
    main()
