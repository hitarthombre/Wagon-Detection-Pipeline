import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

def detect_blur(frame, threshold=100.0):
    """Detect blur using Laplacian variance method"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()
    status = "Clear" if blur_score >= threshold else "Blurry"
    return blur_score, status

def enhance_frame(frame, status, strength=1.1):
    """Enhance frame only if blurry"""
    if status == "Blurry":
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        enhanced = cv2.convertScaleAbs(sharpened, alpha=strength, beta=10)
        return enhanced, True
    else:
        return frame, False

def detect_objects(frame, model, conf_threshold=0.25):
    """
    Detect objects using YOLOv8 model with manual bounding box drawing.
    
    Args:
        frame: Input frame (BGR)
        model: YOLOv8 model instance
        conf_threshold: Confidence threshold for detections
        
    Returns:
        annotated_frame: Frame with manually drawn bounding boxes and confidence scores
        detections: List of detection results
    """
    # Run YOLOv8 inference
    results = model(frame, conf=conf_threshold, verbose=False)
    
    # Create copy for manual annotation
    annotated_frame = frame.copy()
    
    # Extract detection information and draw manually
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract box coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Convert to integers for drawing
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box in green
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare confidence text with 2 decimal places
            conf_text = f"{confidence:.2f}"
            label_text = f"{class_name}: {conf_text}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         (0, 255, 0), -1)
            
            # Draw confidence text above bounding box in black for contrast
            cv2.putText(annotated_frame, label_text, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Store detection info
            detection = {
                'class': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            }
            detections.append(detection)
    
    return annotated_frame, detections

def process_video_with_detection(video_path, model_name='yolov8n.pt', blur_threshold=100.0, 
                                 conf_threshold=0.25):
    """Process video with full pipeline including object detection"""
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    print(f"Model loaded successfully")
    
    # Enable GPU acceleration for OpenCV
    try:
        cv2.ocl.setUseOpenCL(True)
        print(f"OpenCL enabled: {cv2.ocl.useOpenCL()}")
    except:
        print("OpenCL not available")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, Total frames: {total_frames}")
    print(f"Blur threshold: {blur_threshold}, Confidence threshold: {conf_threshold}")
    
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Step 2: Detect blur
            blur_score, blur_status = detect_blur(frame, blur_threshold)
            
            # Step 3: Conditional enhancement
            enhanced_frame, was_enhanced = enhance_frame(frame, blur_status)
            
            # Step 4: Object detection
            detected_frame, detections = detect_objects(enhanced_frame, model, conf_threshold)
            total_detections += len(detections)
            
            # Add info overlay to detected frame
            cv2.putText(detected_frame, f"Frame: {frame_count}/{total_frames}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(detected_frame, f"Detections: {len(detections)}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(detected_frame, f"Blur: {blur_score:.1f} ({blur_status})", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(detected_frame, f"Enhanced: {'Yes' if was_enhanced else 'No'}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Store current frame for pause display
            current_display_frame = detected_frame.copy()
        
        # Add pause indicator if paused
        if paused:
            cv2.putText(current_display_frame, "PAUSED - Press 'r' to resume", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display
        cv2.imshow('Step 4: Object Detection Pipeline', current_display_frame)
        
        # Enhanced controls: p=pause, r=resume, q=quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = True
            print("Video paused. Press 'r' to resume.")
        elif key == ord('r'):
            if paused:
                paused = False
                print("Video resumed.")
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    
    print(f"\nProcessing complete:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per frame: {avg_detections:.2f}")
    print(f"  Processing time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    # Default to video in video folder
    if len(sys.argv) < 2:
        default_video = "video/video_test_1.mp4"
        if os.path.exists(default_video):
            print(f"Using default video: {default_video}")
            video_path = default_video
        else:
            print("Usage: python step4_object_detection.py <video_file_path> [model_name] [blur_threshold] [conf_threshold]")
            print("Example: python step4_object_detection.py video.mp4 yolov8n.pt 100 0.25")
            print("\nControls:")
            print("  'p' - Pause video")
            print("  'r' - Resume video")
            print("  'q' - Quit")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    # Optional parameters
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'yolov8n.pt'
    blur_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    conf_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    
    process_video_with_detection(video_path, model_name, blur_threshold, conf_threshold)
