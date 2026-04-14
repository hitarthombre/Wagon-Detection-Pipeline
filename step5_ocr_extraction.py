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

def preprocess_for_ocr(frame):
    """Enhanced preprocessing for better OCR accuracy.
    
    Args:
        frame: Input frame (BGR)
    
    Returns:
        preprocessed: Preprocessed frame optimized for OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding for better text contrast
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to BGR for EasyOCR
    preprocessed = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
    
    return preprocessed

def perform_ocr(frame, ocr_engine, ocr_engine_type='easyocr', use_preprocessing=True):
    """Perform OCR on frame using selected engine with enhanced preprocessing.
    
    Args:
        frame: Input frame (BGR)
        ocr_engine: OCR engine instance
        ocr_engine_type: Type of engine ('easyocr', 'tesseract', 'paddleocr')
        use_preprocessing: Whether to apply preprocessing for better accuracy
    
    Returns:
        ocr_frame: Frame with OCR results drawn
        text_results: List of detected text with confidence
    """
    ocr_frame = frame.copy()
    text_results = []
    
    try:
        if ocr_engine_type == 'easyocr':
            # Apply preprocessing for better OCR accuracy
            if use_preprocessing:
                processed_frame = preprocess_for_ocr(frame)
                
                # Run OCR on both original and preprocessed, combine results
                results_original = ocr_engine.readtext(frame)
                results_processed = ocr_engine.readtext(processed_frame)
                
                # Combine and deduplicate results (keep higher confidence)
                combined_results = {}
                for bbox, text, confidence in results_original + results_processed:
                    if text not in combined_results or confidence > combined_results[text][1]:
                        combined_results[text] = (bbox, confidence)
                
                results = [(bbox, text, conf) for text, (bbox, conf) in combined_results.items()]
            else:
                results = ocr_engine.readtext(frame)
            
            # Create output frame
            for bbox, text, confidence in results:
                # Convert bbox to integer coordinates
                points = np.array(bbox, dtype=np.int32)
                
                # Draw bounding box in green
                cv2.polylines(ocr_frame, [points], True, (0, 255, 0), 2)
                
                # Prepare text with confidence (2 decimal places)
                label = f"{text} ({confidence:.2f})"
                
                # Calculate text size for background
                text_pos = (int(points[0][0]), int(points[0][1]) - 10)
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw background rectangle for better visibility
                cv2.rectangle(
                    ocr_frame,
                    (text_pos[0], text_pos[1] - text_height - 5),
                    (text_pos[0] + text_width, text_pos[1] + 5),
                    (0, 255, 0), -1
                )
                
                # Draw text in black for contrast
                cv2.putText(ocr_frame, label, text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Store result
                text_results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        elif ocr_engine_type == 'tesseract':
            import pytesseract
            
            # Preprocess for Tesseract
            processed = preprocess_for_ocr(frame) if use_preprocessing else frame
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # Get detailed data with bounding boxes
            config = ocr_engine.get('config', '--psm 6')
            data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 0:
                    text = data['text'][i].strip()
                    if text:
                        confidence = int(conf) / 100.0
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        
                        cv2.rectangle(ocr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        label = f"{text} ({confidence:.2f})"
                        cv2.putText(ocr_frame, label, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        text_results.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        })
        
        elif ocr_engine_type == 'paddleocr':
            # PaddleOCR
            processed = preprocess_for_ocr(frame) if use_preprocessing else frame
            result = ocr_engine.ocr(processed, cls=True)
            
            if result and result[0]:
                for line in result[0]:
                    bbox, (text, confidence) = line
                    points = np.array(bbox, dtype=np.int32)
                    cv2.polylines(ocr_frame, [points], True, (0, 255, 0), 2)
                    
                    label = f"{text} ({confidence:.2f})"
                    text_pos = (int(points[0][0]), int(points[0][1]) - 10)
                    cv2.putText(ocr_frame, label, text_pos,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    text_results.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
    
    except Exception as e:
        print(f"OCR error ({ocr_engine_type}): {e}")
    
    return ocr_frame, text_results

def process_video_with_ocr(video_path, blur_threshold=100.0, ocr_engine='easyocr', use_gpu=True, save_output=True):
    """Process video with full pipeline including OCR
    
    Args:
        video_path: Path to video file
        blur_threshold: Threshold for blur detection
        ocr_engine: OCR engine to use ('easyocr', 'tesseract', 'paddleocr')
        use_gpu: Whether to use GPU (for EasyOCR and PaddleOCR)
        save_output: Whether to save OCR results to file
    """
    # Create output directory
    output_dir = "output"
    if save_output and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Initialize OCR engine
    print(f"Initializing {ocr_engine.upper()} (GPU: {use_gpu})...")
    ocr = None
    ocr_type = ocr_engine.lower()
    
    if ocr_type == 'easyocr':
        import easyocr
        ocr = easyocr.Reader(['en'], gpu=use_gpu)
    elif ocr_type == 'tesseract':
        import pytesseract
        ocr = {'config': '--psm 6'}  # Default config
        try:
            pytesseract.get_tesseract_version()
        except:
            print("Tesseract not found. Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
            return
    elif ocr_type == 'paddleocr':
        from paddleocr import PaddleOCR
        import logging
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
        except:
            try:
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
            except Exception as e:
                print(f"Failed to initialize PaddleOCR: {e}")
                return
    else:
        print(f"Unknown OCR engine: {ocr_engine}")
        return
    
    print(f"{ocr_engine.upper()} initialized successfully")
    
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
    paused = False
    
    # Prepare output file for OCR results
    ocr_output_file = None
    if save_output:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        ocr_output_file = os.path.join(output_dir, f"{video_name}_ocr_results.txt")
        with open(ocr_output_file, 'w', encoding='utf-8') as f:
            f.write(f"OCR Results for: {video_path}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
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
            
            # Step 5: OCR extraction with enhanced preprocessing
            ocr_frame, text_results = perform_ocr(enhanced_frame, ocr, ocr_type, use_preprocessing=True)
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
            
            # Save OCR results to file
            if text_results and save_output:
                with open(ocr_output_file, 'a', encoding='utf-8') as f:
                    f.write(f"Frame {frame_count} - Detected text:\n")
                    for i, result in enumerate(text_results, 1):
                        f.write(f"  {i}. '{result['text']}' (confidence: {result['confidence']:.2f})\n")
                    f.write("\n")
            
            # Display detected text in console
            if text_results:
                print(f"\nFrame {frame_count} - Detected text:")
                for i, result in enumerate(text_results, 1):
                    print(f"  {i}. '{result['text']}' (confidence: {result['confidence']:.2f})")
            
            # Store current frame for pause display
            current_display_frame = ocr_frame.copy()
        
        # Add pause indicator if paused
        if paused:
            cv2.putText(current_display_frame, "PAUSED - Press 'r' to resume", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display
        cv2.imshow('Step 5: OCR Text Extraction', current_display_frame)
        
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
    avg_text_per_frame = total_text_detected / frame_count if frame_count > 0 else 0
    
    print(f"\nProcessing complete:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total text instances detected: {total_text_detected}")
    print(f"  Average text per frame: {avg_text_per_frame:.2f}")
    print(f"  Processing time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    
    if save_output and ocr_output_file:
        # Write summary to output file
        with open(ocr_output_file, 'a', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total frames processed: {frame_count}\n")
            f.write(f"Total text instances detected: {total_text_detected}\n")
            f.write(f"Average text per frame: {avg_text_per_frame:.2f}\n")
            f.write(f"Processing time: {total_time:.2f}s\n")
            f.write(f"Average FPS: {avg_fps:.2f}\n")
        print(f"\n  OCR results saved to: {ocr_output_file}")
    
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
            print("Usage: python step5_ocr_extraction.py <video_file_path> [blur_threshold] [ocr_engine] [use_gpu] [save_output]")
            print("Example: python step5_ocr_extraction.py video.mp4 100 easyocr True True")
            print("\nOCR Engines: easyocr, tesseract, paddleocr")
            print("\nControls:")
            print("  'p' - Pause video")
            print("  'r' - Resume video")
            print("  'q' - Quit")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    # Optional parameters
    blur_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    ocr_engine = sys.argv[3].lower() if len(sys.argv) > 3 else 'easyocr'
    use_gpu = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else True
    save_output = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else True
    
    process_video_with_ocr(video_path, blur_threshold, ocr_engine, use_gpu, save_output)
