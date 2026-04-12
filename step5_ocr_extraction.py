import cv2
import numpy as np
import time
import os
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

def perform_ocr(frame, ocr_engine):
    """
    Perform OCR on frame using EasyOCR.
    
    Args:
        frame: Input frame (BGR)
        ocr_engine: EasyOCR Reader instance
        
    Returns:
        ocr_frame: Frame with OCR results drawn
        text_results: List of detected text with confidence
    """
    # Perform OCR
    results = ocr_engine.readtext(frame)
    
    # Create output frame
    ocr_frame = frame.copy()
    text_results = []
    
    for bbox, text, confidence in results:
        # Convert bbox to integer coordinates
        points = np.array(bbox, dtype=np.int32)
        
        # Draw bounding box
        cv2.polylines(ocr_frame, [points], True, (0, 255, 0), 2)
        
        # Draw text above bounding box
        text_pos = (int(points[0][0]), int(points[0][1]) - 10)
        cv2.putText(ocr_frame, f"{text} ({confidence:.2f})", 
                   text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Store result
        text_results.append({
            'text': text,
            'confidence': confidence,
            'bbox': bbox
        })
    
    return ocr_frame, text_results

def process_video_with_ocr(video_path, blur_threshold=100.0, use_gpu=True):
    """Process video with full pipeline including OCR"""
    
    # Initialize EasyOCR
    print(f"Initializing EasyOCR (GPU: {use_gpu})...")
    ocr = easyocr.Reader(['en'], gpu=use_gpu)
    print("EasyOCR initialized successfully")
    
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
    print(f"Blur threshold: {blur_threshold}")
    
    frame_count = 0
    total_text_detected = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Step 2: Detect blur
        blur_score, blur_status = detect_blur(frame, blur_threshold)
        
        # Step 3: Conditional enhancement
        enhanced_frame, was_enhanced = enhance_frame(frame, blur_status)
        
        # Step 5: OCR extraction
        ocr_frame, text_results = perform_ocr(enhanced_frame, ocr)
        total_text_detected += len(text_results)
        
        # Add info overlay
        cv2.putText(ocr_frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(ocr_frame, f"Blur: {blur_score:.1f} ({blur_status})", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ocr_frame, f"Enhanced: {'Yes' if was_enhanced else 'No'}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ocr_frame, f"Text Detected: {len(text_results)}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display detected text in console
        if text_results:
            print(f"\nFrame {frame_count} - Detected text:")
            for i, result in enumerate(text_results, 1):
                print(f"  {i}. '{result['text']}' (confidence: {result['confidence']:.2f})")
        
        # Display
        cv2.imshow('Step 5: OCR Text Extraction', ocr_frame)
        
        # Control (q to quit, p to pause)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(-1)
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_text_per_frame = total_text_detected / frame_count if frame_count > 0 else 0
    
    print(f"\nProcessing complete:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total text instances detected: {total_text_detected}")
    print(f"  Average text per frame: {avg_text_per_frame:.2f}")
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
            print("Usage: python step5_ocr_extraction.py <video_file_path> [blur_threshold] [use_gpu]")
            print("Example: python step5_ocr_extraction.py video.mp4 100 True")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    # Optional parameters
    blur_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    use_gpu = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else True
    
    process_video_with_ocr(video_path, blur_threshold, use_gpu)
