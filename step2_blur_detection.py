import cv2
import time
import os

def detect_blur(frame, threshold=100.0):
    """
    Detect blur in a frame using Laplacian variance method.
    
    Args:
        frame: Input frame (BGR)
        threshold: Blur threshold (default 100.0)
        
    Returns:
        blur_score: Variance of Laplacian
        status: "Clear" or "Blurry"
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()
    
    # Classify based on threshold
    status = "Clear" if blur_score >= threshold else "Blurry"
    
    return blur_score, status

def process_video_blur_detection(video_path, blur_threshold=100.0):
    """
    Process video and detect blur in each frame.
    """
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
    blurry_count = 0
    clear_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Detect blur
        blur_score, status = detect_blur(frame, blur_threshold)
        
        # Update counters
        if status == "Blurry":
            blurry_count += 1
            color = (0, 0, 255)  # Red for blurry
        else:
            clear_count += 1
            color = (0, 255, 0)  # Green for clear
        
        # Add overlay
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Blur Score: {blur_score:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Clear: {clear_count} | Blurry: {blurry_count}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Step 2: Blur Detection', frame)
        
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
    print(f"  Clear frames: {clear_count} ({clear_count/frame_count*100:.1f}%)")
    print(f"  Blurry frames: {blurry_count} ({blurry_count/frame_count*100:.1f}%)")
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
            print("Usage: python step2_blur_detection.py <video_file_path> [blur_threshold]")
            print("Example: python step2_blur_detection.py video.mp4 100")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    # Optional blur threshold parameter
    blur_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    
    process_video_blur_detection(video_path, blur_threshold)
