"""
OpenCV Dashboard for AI Video Processing Pipeline - Export Version
Saves processed video with all stages visualized in a grid layout
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
    
    banner_height = 40
    if position == 'top':
        cv2.rectangle(labeled_frame, (0, 0), (w, banner_height), bg_color, -1)
        text_y = 28
    else:
        cv2.rectangle(labeled_frame, (0, h - banner_height), (w, h), bg_color, -1)
        text_y = h - 12
    
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
    
    cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
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
    
    target_h, target_w = frames[0].shape[:2]
    resized_frames = []
    
    for frame, label in zip(frames, labels):
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h))
        
        labeled_frame = add_label(frame, label, position='top')
        resized_frames.append(labeled_frame)
    
    total_slots = rows * cols
    while len(resized_frames) < total_slots:
        black_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        resized_frames.append(black_frame)
    
    grid_rows = []
    for i in range(rows):
        row_frames = resized_frames[i * cols:(i + 1) * cols]
        grid_row = np.hstack(row_frames)
        grid_rows.append(grid_row)
    
    grid = np.vstack(grid_rows)
    return grid

def process_video_pipeline(video_path, output_path="output_dashboard.mp4", 
                           blur_threshold=100.0, ocr_confidence=0.7, 
                           enable_ocr=True, display_size=(640, 480)):
    """Process video and export dashboard view"""
    
    print("=" * 70)
    print("AI VIDEO PROCESSING PIPELINE - DASHBOARD EXPORT")
    print("=" * 70)
    print(f"Input Video: {video_path}")
    print(f"Output Video: {output_path}")
    print(f"Blur Threshold: {blur_threshold}")
    print(f"OCR Confidence: {ocr_confidence}")
    print(f"Display Size: {display_size}")
    print(f"OCR Enabled: {enable_ocr}")
    
    # Initialize OCR
    ocr_engine = None
    if enable_ocr:
        print("\nInitializing EasyOCR...")
        ocr_engine = easyocr.Reader(['en'], gpu=True)
        print("✓ EasyOCR initialized")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Error: Cannot open video {video_path}")
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
    
    # Setup video writer
    grid_width = display_size[0] * 2
    grid_height = display_size[1] * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
    
    print(f"\nOutput Grid Size: {grid_width}x{grid_height}")
    print("\nProcessing...")
    print("-" * 70)
    
    # Statistics
    frame_count = 0
    blurry_count = 0
    clear_count = 0
    enhanced_count = 0
    total_text_detected = 0
    start_time = time.time()
    last_print_time = start_time
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Resize frame
        display_frame = cv2.resize(frame, display_size)
        
        # Step 1: Original
        original_frame = display_frame.copy()
        
        # Step 2: Blur Detection
        blur_score, blur_status = detect_blur(display_frame, blur_threshold)
        if blur_status == "Blurry":
            blurry_count += 1
            status_color = (0, 0, 255)
        else:
            clear_count += 1
            status_color = (0, 255, 0)
        
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
            enhance_color = (0, 165, 255)
        else:
            enhance_text = "ORIGINAL"
            enhance_color = (0, 255, 0)
        
        cv2.putText(enhanced_frame, enhance_text, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, enhance_color, 2)
        
        # Step 4: OCR
        if enable_ocr and ocr_engine:
            ocr_frame, text_results = perform_ocr(enhanced_frame, ocr_engine, ocr_confidence)
            total_text_detected += len(text_results)
            cv2.putText(ocr_frame, f"Text: {len(text_results)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            ocr_frame = enhanced_frame.copy()
            cv2.putText(ocr_frame, "OCR Disabled", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            text_results = []
        
        # Create grid
        frames = [original_frame, blur_frame, enhanced_frame, ocr_frame]
        labels = [
            "Step 1: Original",
            "Step 2: Blur Detection",
            "Step 3: Enhancement",
            "Step 4: OCR"
        ]
        
        grid = create_grid_layout(frames, labels, grid_size=(2, 2))
        
        # Add stats
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
        
        # Write frame
        out.write(grid)
        
        # Progress update every 2 seconds
        if current_time - last_print_time >= 2.0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {processing_fps:.1f}")
            last_print_time = current_time
    
    # Cleanup
    cap.release()
    out.release()
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output saved to: {output_path}")
    print(f"\nStatistics:")
    print(f"  Total Frames: {frame_count}")
    print(f"  Processing Time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"\nBlur Detection:")
    print(f"  Clear: {clear_count} ({clear_count/frame_count*100:.1f}%)")
    print(f"  Blurry: {blurry_count} ({blurry_count/frame_count*100:.1f}%)")
    print(f"\nEnhancement:")
    print(f"  Enhanced: {enhanced_count}")
    print(f"  Efficiency: {(frame_count-enhanced_count)/frame_count*100:.1f}%")
    if enable_ocr:
        print(f"\nOCR:")
        print(f"  Total Text: {total_text_detected}")
        print(f"  Avg/Frame: {total_text_detected/frame_count:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    
    default_video = "video/video_test_1.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    elif Path(default_video).exists():
        video_path = default_video
    else:
        print("Usage: python opencv_dashboard_export.py <video> [output] [blur_th] [ocr_conf] [enable_ocr]")
        sys.exit(1)
    
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output_dashboard.mp4"
    blur_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    ocr_confidence = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
    enable_ocr = sys.argv[5].lower() != 'false' if len(sys.argv) > 5 else False
    
    process_video_pipeline(
        video_path=video_path,
        output_path=output_path,
        blur_threshold=blur_threshold,
        ocr_confidence=ocr_confidence,
        enable_ocr=enable_ocr,
        display_size=(640, 480)
    )
