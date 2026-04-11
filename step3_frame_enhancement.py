import cv2
import numpy as np
import time
import os

def detect_blur(frame, threshold=100.0):
    """Detect blur using Laplacian variance method"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()
    status = "Clear" if blur_score >= threshold else "Blurry"
    return blur_score, status

def enhance_frame(frame, status):
    """
    Enhance frame only if blurry, otherwise return original.
    
    Args:
        frame: Input frame (BGR)
        status: "Clear" or "Blurry"
        
    Returns:
        enhanced_frame: Enhanced frame if blurry, original if clear
        was_enhanced: Boolean indicating if enhancement was applied
    """
    if status == "Blurry":
        # Apply sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        
        # Increase brightness slightly
        enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
        
        return enhanced, True
    else:
        # Return original frame without processing
        return frame, False

def process_video_with_enhancement(video_path, blur_threshold=100.0):
    """Process video with blur detection and conditional enhancement"""
    # Enable GPU acceleration
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
    enhanced_count = 0
    skipped_count = 0
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
        
        if was_enhanced:
            enhanced_count += 1
            status_color = (0, 165, 255)  # Orange for enhanced
            status_text = "ENHANCED"
        else:
            skipped_count += 1
            status_color = (0, 255, 0)  # Green for skipped
            status_text = "ORIGINAL"
        
        # Create side-by-side comparison
        comparison = np.hstack([frame, enhanced_frame])
        
        # Add labels
        cv2.putText(comparison, "Original", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Processed", 
                    (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add status info
        cv2.putText(comparison, f"Frame: {frame_count}/{total_frames}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, f"Blur: {blur_score:.1f} ({blur_status})", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, f"Action: {status_text}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(comparison, f"Enhanced: {enhanced_count} | Skipped: {skipped_count}", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('Step 3: Conditional Frame Enhancement', comparison)
        
        # Control (q to quit, p to pause)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(-1)
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nProcessing complete:")
    print(f"  Total frames: {frame_count}")
    print(f"  Enhanced frames: {enhanced_count} ({enhanced_count/frame_count*100:.1f}%)")
    print(f"  Skipped frames: {skipped_count} ({skipped_count/frame_count*100:.1f}%)")
    print(f"  Processing time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Performance gain: {skipped_count/frame_count*100:.1f}% frames skipped enhancement")
    
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
            print("Usage: python step3_frame_enhancement.py <video_file_path> [blur_threshold]")
            print("Example: python step3_frame_enhancement.py video.mp4 100")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    # Optional blur threshold parameter
    blur_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    
    process_video_with_enhancement(video_path, blur_threshold)
