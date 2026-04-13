import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
import easyocr

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
    """
    Detect objects using YOLOv8 with manual bounding box drawing.
    Shows confidence scores with 2 decimal places in green.
    """
    results = model(frame, conf=conf_threshold, verbose=False)
    annotated_frame = frame.copy()
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw green bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Format confidence with 2 decimals
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text background size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw green background for text
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         (0, 255, 0), -1)
            
            # Draw text in black for contrast
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

def preprocess_for_ocr(frame):
    """Enhanced preprocessing for better OCR accuracy"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    preprocessed = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
    return preprocessed

def perform_ocr_enhanced(frame, ocr_engine, use_preprocessing=True):
    """Perform OCR with enhanced preprocessing for better accuracy"""
    if use_preprocessing:
        processed_frame = preprocess_for_ocr(frame)
        results_original = ocr_engine.readtext(frame)
        results_processed = ocr_engine.readtext(processed_frame)
        
        # Combine and deduplicate
        combined_results = {}
        for bbox, text, confidence in results_original + results_processed:
            if text not in combined_results or confidence > combined_results[text][1]:
                combined_results[text] = (bbox, confidence)
        
        results = [(bbox, text, conf) for text, (bbox, conf) in combined_results.items()]
    else:
        results = ocr_engine.readtext(frame)
    
    ocr_frame = frame.copy()
    text_results = []
    
    for bbox, text, confidence in results:
        points = np.array(bbox, dtype=np.int32)
        cv2.polylines(ocr_frame, [points], True, (0, 255, 0), 2)
        
        label = f"{text} ({confidence:.2f})"
        text_pos = (int(points[0][0]), int(points[0][1]) - 10)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Green background for text
        cv2.rectangle(ocr_frame,
                     (text_pos[0], text_pos[1] - text_height - 5),
                     (text_pos[0] + text_width, text_pos[1] + 5),
                     (0, 255, 0), -1)
        
        cv2.putText(ocr_frame, label, text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        text_results.append({
            'text': text,
            'confidence': confidence,
            'bbox': bbox
        })
    
    return ocr_frame, text_results

def process_integrated_pipeline_headless(video_path, model_name='yolov8n.pt', 
                                        blur_threshold=100.0, conf_threshold=0.25,
                                        enable_ocr=True, use_gpu=True, save_output=True,
                                        save_video=True):
    """
    Headless pipeline that saves output video instead of displaying.
    Perfect for environments without GUI support.
    """
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Load models
    print(f"Loading YOLOv8 model: {model_name}")
    yolo_model = YOLO(model_name)
    print("YOLOv8 model loaded")
    
    ocr_engine = None
    if enable_ocr:
        print(f"Initializing EasyOCR (GPU: {use_gpu})...")
        ocr_engine = easyocr.Reader(['en'], gpu=use_gpu)
        print("EasyOCR initialized")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {width}x{height} @ {fps:.2f} FPS, Total frames: {total_frames}")
    print(f"Blur threshold: {blur_threshold}, Confidence threshold: {conf_threshold}")
    print(f"OCR enabled: {enable_ocr}, Save output: {save_output}, Save video: {save_video}\n")
    
    # Prepare output video
    video_writer = None
    if save_video:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"{video_name}_processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_video_path}")
    
    # Prepare output files
    detection_output = None
    ocr_output = None
    
    if save_output:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        detection_output = os.path.join(output_dir, f"{video_name}_detections.txt")
        
        with open(detection_output, 'w', encoding='utf-8') as f:
            f.write(f"Detection Results for: {video_path}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        
        if enable_ocr:
            ocr_output = os.path.join(output_dir, f"{video_name}_ocr_results.txt")
            with open(ocr_output, 'w', encoding='utf-8') as f:
                f.write(f"OCR Results for: {video_path}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
    
    frame_count = 0
    total_detections = 0
    total_text_detected = 0
    start_time = time.time()
    last_progress_time = start_time
    
    print("Processing video...")
    print("Progress: [", end="", flush=True)
    progress_bar_length = 50
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Progress bar
        progress = frame_count / total_frames
        if time.time() - last_progress_time > 0.5:  # Update every 0.5 seconds
            filled = int(progress_bar_length * progress)
            bar = "=" * filled + " " * (progress_bar_length - filled)
            print(f"\rProgress: [{bar}] {progress*100:.1f}% ({frame_count}/{total_frames})", end="", flush=True)
            last_progress_time = time.time()
        
        # Step 2: Blur detection
        blur_score, blur_status = detect_blur(frame, blur_threshold)
        
        # Step 3: Enhancement
        enhanced_frame, was_enhanced = enhance_frame(frame, blur_status)
        
        # Step 4: Object detection with manual drawing
        detected_frame, detections = detect_objects_manual(
            enhanced_frame, yolo_model, conf_threshold
        )
        total_detections += len(detections)
        
        # Save detection results
        if detections and save_output:
            with open(detection_output, 'a', encoding='utf-8') as f:
                f.write(f"Frame {frame_count} - Detections: {len(detections)}\n")
                for i, det in enumerate(detections, 1):
                    f.write(f"  {i}. {det['class_name']}: {det['confidence']:.2f} "
                           f"at {det['bbox']}\n")
                f.write("\n")
        
        # Step 5: OCR (if enabled)
        if enable_ocr and ocr_engine:
            ocr_frame, text_results = perform_ocr_enhanced(
                detected_frame, ocr_engine, use_preprocessing=True
            )
            total_text_detected += len(text_results)
            
            # Save OCR results
            if text_results and save_output:
                with open(ocr_output, 'a', encoding='utf-8') as f:
                    f.write(f"Frame {frame_count} - Text detected: {len(text_results)}\n")
                    for i, result in enumerate(text_results, 1):
                        f.write(f"  {i}. '{result['text']}' "
                               f"(confidence: {result['confidence']:.2f})\n")
                    f.write("\n")
            
            final_frame = ocr_frame
        else:
            final_frame = detected_frame
        
        # Add info overlay
        cv2.putText(final_frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(final_frame, f"Detections: {len(detections)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(final_frame, f"Blur: {blur_score:.1f} ({blur_status})", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(final_frame, f"Enhanced: {'Yes' if was_enhanced else 'No'}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if enable_ocr:
            cv2.putText(final_frame, f"Text: {len(text_results) if 'text_results' in locals() else 0}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to output video
        if video_writer is not None:
            video_writer.write(final_frame)
    
    print("]")  # Close progress bar
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    avg_text = total_text_detected / frame_count if frame_count > 0 else 0
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total detections: {total_detections} (avg: {avg_detections:.2f} per frame)")
    if enable_ocr:
        print(f"Total text instances: {total_text_detected} (avg: {avg_text:.2f} per frame)")
    print(f"Processing time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    
    if save_output:
        # Write summaries
        with open(detection_output, 'a', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total frames: {frame_count}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Average detections per frame: {avg_detections:.2f}\n")
            f.write(f"Processing time: {total_time:.2f}s\n")
            f.write(f"Average FPS: {avg_fps:.2f}\n")
        
        print(f"\nDetection results saved to: {detection_output}")
        
        if enable_ocr and ocr_output:
            with open(ocr_output, 'a', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("SUMMARY\n")
                f.write("="*80 + "\n")
                f.write(f"Total frames: {frame_count}\n")
                f.write(f"Total text instances: {total_text_detected}\n")
                f.write(f"Average text per frame: {avg_text:.2f}\n")
                f.write(f"Processing time: {total_time:.2f}s\n")
                f.write(f"Average FPS: {avg_fps:.2f}\n")
            
            print(f"OCR results saved to: {ocr_output}")
    
    if save_video:
        print(f"Processed video saved to: {output_video_path}")
        print(f"\nYou can view the video with confidence scores using any video player.")

if __name__ == "__main__":
    import sys
    
    # Default to video in video folder
    if len(sys.argv) < 2:
        default_video = "video/video_test_1.mp4"
        if os.path.exists(default_video):
            print(f"Using default video: {default_video}")
            video_path = default_video
        else:
            print("Usage: python integrated_pipeline_headless.py <video_file_path> [options]")
            print("\nOptions:")
            print("  model_name (default: yolov8n.pt)")
            print("  blur_threshold (default: 100.0)")
            print("  conf_threshold (default: 0.25)")
            print("  enable_ocr (default: True)")
            print("  use_gpu (default: True)")
            print("  save_output (default: True)")
            print("  save_video (default: True)")
            print("\nExample:")
            print("  python integrated_pipeline_headless.py video.mp4 yolov8n.pt 100 0.25 True True True True")
            print("\nNote: This is a headless version that saves output video instead of displaying.")
            print("      Perfect for environments without GUI support.")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    # Optional parameters
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'yolov8n.pt'
    blur_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    conf_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    enable_ocr = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else True
    use_gpu = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else True
    save_output = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else True
    save_video = sys.argv[8].lower() == 'true' if len(sys.argv) > 8 else True
    
    process_integrated_pipeline_headless(video_path, model_name, blur_threshold, 
                                        conf_threshold, enable_ocr, use_gpu, 
                                        save_output, save_video)
