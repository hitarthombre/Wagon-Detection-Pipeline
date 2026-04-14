"""
Quick video processing without OCR for fast results.
Use this for rapid testing and demos.
"""

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

def detect_objects_manual(frame, model, conf_threshold=0.25):
    """Detect objects with manual confidence display"""
    results = model(frame, conf=conf_threshold, verbose=False)
    annotated_frame = frame.copy()
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw green bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Format confidence with 2 decimals
            label = f"{class_name}: {confidence:.2f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw green background
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         (0, 255, 0), -1)
            
            # Draw text in black
            cv2.putText(annotated_frame, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            detections.append({
                'class': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
    
    return annotated_frame, detections

def quick_process(video_path, model_name='yolov8n.pt', 
                 blur_threshold=100.0, conf_threshold=0.25,
                 save_video=True, save_results=True):
    """
    Fast processing without OCR.
    Perfect for quick demos and testing.
    """
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load YOLO model
    print(f"Loading YOLOv8 model: {model_name}")
    yolo_model = YOLO(model_name)
    print("Model loaded\n")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Confidence threshold: {conf_threshold}\n")
    
    # Prepare output video
    video_writer = None
    if save_video:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"{video_name}_processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Saving to: {output_video_path}\n")
    
    # Prepare results file
    results_file = None
    if save_results:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        results_file = os.path.join(output_dir, f"{video_name}_detections.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"Detection Results for: {video_path}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    last_update = start_time
    
    print("Processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Progress update every second
        if time.time() - last_update > 1.0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
                  f"FPS: {fps_current:.1f} | ETA: {eta:.0f}s", end="\r")
            last_update = time.time()
        
        # Blur detection
        blur_score, blur_status = detect_blur(frame, blur_threshold)
        
        # Enhancement
        enhanced_frame, was_enhanced = enhance_frame(frame, blur_status)
        
        # Object detection
        detected_frame, detections = detect_objects_manual(
            enhanced_frame, yolo_model, conf_threshold
        )
        total_detections += len(detections)
        
        # Save detections
        if detections and save_results:
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"Frame {frame_count} - Detections: {len(detections)}\n")
                for i, det in enumerate(detections, 1):
                    f.write(f"  {i}. {det['class_name']}: {det['confidence']:.2f} "
                           f"at {det['bbox']}\n")
                f.write("\n")
        
        # Add overlay
        cv2.putText(detected_frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(detected_frame, f"Detections: {len(detections)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(detected_frame, f"Blur: {blur_score:.1f} ({blur_status})", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write to video
        if video_writer is not None:
            video_writer.write(detected_frame)
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    
    print(f"\n\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {total_detections} (avg: {avg_detections:.2f} per frame)")
    print(f"Processing time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    
    if save_results:
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total frames: {frame_count}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Average detections per frame: {avg_detections:.2f}\n")
            f.write(f"Processing time: {total_time:.2f}s\n")
            f.write(f"Average FPS: {avg_fps:.2f}\n")
        print(f"\nResults saved to: {results_file}")
    
    if save_video:
        print(f"Video saved to: {output_video_path}")
        print(f"\nView the video to see confidence scores on detections!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        default_video = "video/video_test_1.mp4"
        if os.path.exists(default_video):
            print(f"Using default video: {default_video}\n")
            video_path = default_video
        else:
            print("Usage: python quick_process.py <video_path> [model] [blur_threshold] [conf_threshold]")
            print("\nExample:")
            print("  python quick_process.py video/video_test_1.mp4")
            print("  python quick_process.py video/video_test_1.mp4 yolov8n.pt 100 0.30")
            print("\nThis script processes video quickly WITHOUT OCR.")
            print("For OCR processing, use: python integrated_pipeline_headless.py")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'yolov8n.pt'
    blur_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    conf_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    
    quick_process(video_path, model_name, blur_threshold, conf_threshold)
