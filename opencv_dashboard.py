"""
OpenCV Dashboard for AI Video Processing Pipeline
Displays all stages: Original → Blur Detection → Enhancement → OCR
"""

import cv2
import numpy as np
import time
import easyocr
from pathlib import Path

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
        return frame.copy(), False

def perform_ocr(frame, ocr_engine, confidence_threshold=0.7):
    """Perform OCR on frame using EasyOCR"""
    results = ocr_engine.readtext(frame)
    
    ocr_frame = frame.copy()
    text_results = []
    
    for bbox, text, confidence in results:
        if confidence >= confidence_threshold:
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(ocr_frame, [points], True, (0, 255, 0), 2)
            text_pos = (int(points[0][0]), int(points[0][1]) - 10)
            cv2.putText(ocr_frame, f"{text} ({confidence:.2f})", 
                       text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            text_results.append({'text': text, 'confidence': confidence})
    
    return ocr_frame, text_results

def add_label(frame, label, position='top', bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """Add a label banner to the frame"""
    h, w = frame.shape[:2]
    labeled_frame = frame.copy()
    
    # Create label banner
    banner_height = 40
    if position == 'top':
        cv2.rectangle(labeled_frame, (0, 0), (w, banner_height), bg_color, -1)
        text_y = 28
    else:  # bottom
        cv2.rectangle(labeled_frame, (0, h - banner_height), (w, h), bg_color, -1)
        text_y = h - 12
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    
    cv2.putText(labeled_frame, label, (text_x, text_y), 
                font, font_scale, text_color, thickness)
    
    return labeled_frame

def add_stats_overlay(frame, stats):
    """Add statistics overlay to frame"""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Semi-transparent background
    cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add stats text
    y_offset = 35
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (20, y_offset), font, font_scale, color, thickness)
        y_offset += 25
    
    return frame

def create_grid_layout(frames, labels, grid_size=(2, 2)):
    """Create a grid layout from multiple frames"""
    rows, cols = grid_size
    
    # Ensure all frames have the same size
    target_h, target_w = frames[0].shape[:2]
    resized_frames = []
    
    for i, (frame, label) in enumerate(zip(frames, labels)):
        # Resize frame if needed
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h))
        
        # Add label
        labeled_frame = add_label(frame, label, position='top')
        resized_frames.append(labeled_frame)
    
    # Fill remaining slots with black frames if needed
    total_slots = rows * cols
    while len(resized_frames) < total_slots:
        black_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        resized_frames.append(black_frame)
    
    # Create grid
    grid_rows = []
    for i in range(rows):
        row_frames = resized_frames[i * cols:(i + 1) * cols]
        grid_row = np.hstack(row_frames)
        grid_rows.append(grid_row)
    
    grid = np.vstack(grid_rows)
    return grid

def process_video_pipeline(video_path, blur_threshold=100.0, ocr_confidence=0.7, 
                           enable_ocr=True, display_size=(640, 480)):
    """Process video through complete pipeline with OpenCV dashboard"""
    
    print("Initializing AI Pipeline Dashboard...")
    print(f"Video: {video_path}")
    print(f"Blur Threshold: {blur_threshold}")
    print(f"OCR Confidence: {ocr_confidence}")
    print(f"Display Size: {display_size}")
    
    # Initialize OCR
    ocr_engine = None
    if enable_ocr:
        print("\nInitializing EasyOCR (this may take a moment)...")
        ocr_engine = easyocr.Reader(['en'], gpu=True)
        print("EasyOCR initialized successfully!")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    
    # Statistics
    frame_count = 0
    blurry_count = 0
    clear_count = 0
    enhanced_count = 0
    total_text_detected = 0
    start_time = time.time()
    
    print("\nProcessing video... Press 'q' to quit, 'p' to pause")
    print("-" * 60)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Resize frame for display
        display_frame = cv2.resize(frame, display_size)
        
        # Step 1: Original Frame
        original_frame = display_frame.copy()
        
        # Step 2: Blur Detection
        blur_score, blur_status = detect_blur(display_frame, blur_threshold)
        if blur_status == "Blurry":
            blurry_count += 1
            status_color = (0, 0, 255)  # Red
        else:
            clear_count += 1
            status_color = (0, 255, 0)  # Green
        
        blur_frame = display_frame.copy()
        cv2.putText(blur_frame, f"Score: {blur_score:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(blur_frame, f"Status: {blur_status}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Step 3: Enhancement
        enhanced_frame, was_enhanced = enhance_frame(display_frame, blur_status)
        if was_enhanced:
            enhanced_count += 1
            enhance_text = "ENHANCED"
            enhance_color = (0, 165, 255)  # Orange
        else:
            enhance_text = "ORIGINAL"
            enhance_color = (0, 255, 0)  # Green
        
        cv2.putText(enhanced_frame, enhance_text, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, enhance_color, 2)
        
        # Step 4: OCR
        if enable_ocr and ocr_engine:
            ocr_frame, text_results = perform_ocr(enhanced_frame, ocr_engine, ocr_confidence)
            total_text_detected += len(text_results)
            
            cv2.putText(ocr_frame, f"Text Found: {len(text_results)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            ocr_frame = enhanced_frame.copy()
            cv2.putText(ocr_frame, "OCR Disabled", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            text_results = []
        
        # Create grid layout
        frames = [original_frame, blur_frame, enhanced_frame, ocr_frame]
        labels = [
            "Step 1: Original Frame",
            "Step 2: Blur Detection",
            "Step 3: Enhancement",
            "Step 4: OCR Extraction"
        ]
        
        grid = create_grid_layout(frames, labels, grid_size=(2, 2))
        
        # Add global statistics overlay
        current_time = time.time()
        elapsed = current_time - start_time
        processing_fps = frame_count / elapsed if elapsed > 0 else 0
        
        stats = {
            'Frame': f"{frame_count}/{total_frames}",
            'FPS': f"{processing_fps:.1f}",
            'Clear': clear_count,
            'Blurry': blurry_count,
            'Enhanced': enhanced_count,
            'Text': total_text_detected
        }
        
        grid = add_stats_overlay(grid, stats)
        
        # Display grid
        cv2.imshow('AI Video Processing Pipeline - Dashboard', grid)
        
        # Control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nStopping...")
            break
        elif key == ord('p'):
            print("\nPaused. Press any key to continue...")
            cv2.waitKey(-1)
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total Frames Processed: {frame_count}")
    print(f"Processing Time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"\nBlur Detection:")
    print(f"  Clear Frames: {clear_count} ({clear_count/frame_count*100:.1f}%)")
    print(f"  Blurry Frames: {blurry_count} ({blurry_count/frame_count*100:.1f}%)")
    print(f"\nEnhancement:")
    print(f"  Enhanced: {enhanced_count}")
    print(f"  Skipped: {frame_count - enhanced_count}")
    print(f"  Efficiency: {(frame_count - enhanced_count)/frame_count*100:.1f}%")
    if enable_ocr:
        print(f"\nOCR:")
        print(f"  Total Text Detected: {total_text_detected}")
        print(f"  Avg per Frame: {total_text_detected/frame_count:.2f}")
    print("=" * 60)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    # Default video path
    default_video = "video/video_test_1.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    elif Path(default_video).exists():
        video_path = default_video
        print(f"Using default video: {default_video}")
    else:
        print("Usage: python opencv_dashboard.py <video_path> [blur_threshold] [ocr_confidence]")
        print("Example: python opencv_dashboard.py video.mp4 100 0.7")
        sys.exit(1)
    
    # Optional parameters
    blur_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    ocr_confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    enable_ocr = sys.argv[4].lower() != 'false' if len(sys.argv) > 4 else True
    
    # Run pipeline
    process_video_pipeline(
        video_path=video_path,
        blur_threshold=blur_threshold,
        ocr_confidence=ocr_confidence,
        enable_ocr=enable_ocr,
        display_size=(640, 480)
    )
