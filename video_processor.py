import cv2
import time

def process_video(video_path):
    """
    Read and display video frames in real-time with frame counter and FPS.
    Uses GPU acceleration if available.
    """
    # Try to use GPU acceleration (Intel iGPU via OpenCL backend)
    try:
        cv2.ocl.setUseOpenCL(True)
        print(f"OpenCL available: {cv2.ocl.haveOpenCL()}")
        print(f"OpenCL enabled: {cv2.ocl.useOpenCL()}")
    except:
        print("OpenCL not available, using CPU")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, Total frames: {total_frames}")
    
    frame_count = 0
    start_time = time.time()
    prev_time = start_time
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or cannot read frame")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        elapsed = current_time - prev_time
        if elapsed > 0:
            current_fps = 1.0 / elapsed
        else:
            current_fps = 0
        prev_time = current_time
        
        # Add frame counter and FPS overlay
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {current_fps:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Video Processing Pipeline - Step 1', frame)
        
        # Control playback speed (press 'q' to quit, 'p' to pause)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(-1)  # Wait until any key is pressed
    
    # Cleanup
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nProcessing complete:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    import os
    
    # Default to video in video folder if no argument provided
    if len(sys.argv) < 2:
        default_video = "video/video_test_1.mp4"
        if os.path.exists(default_video):
            print(f"Using default video: {default_video}")
            video_path = default_video
        else:
            print("Usage: python video_processor.py <video_file_path>")
            print("Example: python video_processor.py sample.mp4")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    process_video(video_path)
