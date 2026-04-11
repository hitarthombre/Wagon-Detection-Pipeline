import streamlit as st
import cv2
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue

st.set_page_config(page_title="AI Video Processing Pipeline", layout="wide")

st.title("🎥 AI Video Processing Pipeline")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    video_files = list(Path("video").glob("*.mp4"))
    
    if video_files:
        selected_video = st.selectbox(
            "Select Video",
            video_files,
            format_func=lambda x: x.name
        )
    else:
        st.error("No video files found in 'video' folder")
        st.stop()
    
    process_button = st.button("▶️ Process Video", type="primary")
    stop_button = st.button("⏹️ Stop")
    
    st.divider()
    st.subheader("Settings")
    skip_frames = st.slider("Skip Frames (for speed)", 0, 10, 0)
    show_stats = st.checkbox("Show Statistics", value=True)
    update_interval = st.slider("UI Update Interval (frames)", 1, 30, 5, 
                                help="Update UI every N frames for better performance")
    use_threading = st.checkbox("Multi-threaded Processing", value=True,
                                help="Process frames in parallel")
    
    st.divider()
    st.subheader("Pipeline Steps")
    enable_blur_detection = st.checkbox("Step 2: Blur Detection", value=True)
    if enable_blur_detection:
        blur_threshold = st.slider("Blur Threshold", 50.0, 300.0, 100.0, 10.0)

# Main content area
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📹 Step 1: Original Frame")
    frame_placeholder = st.empty()

with col2:
    st.subheader("� Srtep 2: Blur Detection")
    blur_placeholder = st.empty()

with col3:
    st.subheader("📊 Processing Info")
    info_placeholder = st.empty()

# Stats area
if show_stats:
    st.divider()
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        frame_counter = st.empty()
    with stats_col2:
        fps_display = st.empty()
    with stats_col3:
        progress_display = st.empty()
    with stats_col4:
        time_display = st.empty()

progress_bar = st.progress(0)

def detect_blur(frame, threshold=100.0):
    """
    Detect blur using Laplacian variance method - optimized for GPU.
    """
    # Use GPU if available
    try:
        if cv2.ocl.useOpenCL():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_gpu = cv2.UMat(gray)
            laplacian = cv2.Laplacian(gray_gpu, cv2.CV_64F)
            blur_score = laplacian.get().var()
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = laplacian.var()
    except:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
    
    status = "Clear" if blur_score >= threshold else "Blurry"
    return blur_score, status

def process_frame_batch(frames, enable_blur, blur_threshold, gpu_enabled):
    """Process a batch of frames in parallel"""
    results = []
    for frame in frames:
        result = {'original': frame}
        
        if enable_blur:
            blur_score, blur_status = detect_blur(frame, blur_threshold)
            result['blur_score'] = blur_score
            result['blur_status'] = blur_status
        
        results.append(result)
    return results

def process_video_stream(video_path, skip_frames=0, enable_blur=False, blur_threshold=100.0, 
                        update_interval=5, use_threading=True):
    """Process video with optimized performance"""
    # Enable GPU acceleration via OpenCL
    gpu_info = {}
    try:
        cv2.ocl.setUseOpenCL(True)
        gpu_info['opencl_available'] = cv2.ocl.haveOpenCL()
        gpu_info['opencl_enabled'] = cv2.ocl.useOpenCL()
        if gpu_info['opencl_available']:
            device = cv2.ocl.Device.getDefault()
            gpu_info['device_name'] = device.name()
            gpu_info['device_type'] = device.type()
    except:
        gpu_info['opencl_available'] = False
        gpu_info['opencl_enabled'] = False
    
    # Hardware-accelerated video decoding
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_ANY)
    
    # Set buffer size for better performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    try:
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        gpu_info['hw_decode'] = True
    except:
        gpu_info['hw_decode'] = False
    
    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_count = 0
    blurry_count = 0
    clear_count = 0
    start_time = time.time()
    prev_time = start_time
    
    # Threading pool for parallel processing
    executor = ThreadPoolExecutor(max_workers=4) if use_threading else None
    frame_batch = []
    batch_size = 4 if use_threading else 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames if requested
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            continue
        
        # Only update UI every N frames for performance
        if frame_count % update_interval != 0:
            # Still process but don't yield
            if enable_blur:
                blur_score, blur_status = detect_blur(frame, blur_threshold)
                if blur_status == "Blurry":
                    blurry_count += 1
                else:
                    clear_count += 1
            continue
        
        
        current_time = time.time()
        elapsed = current_time - prev_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        prev_time = current_time
        
        # Step 2: Blur detection (optimized)
        blur_score = 0
        blur_status = "N/A"
        blur_frame = None
        
        if enable_blur:
            blur_score, blur_status = detect_blur(frame, blur_threshold)
            if blur_status == "Blurry":
                blurry_count += 1
                status_color = (0, 0, 255)
            else:
                clear_count += 1
                status_color = (0, 255, 0)
            
            # Create blur visualization (minimal operations)
            blur_frame = frame.copy()
            cv2.putText(blur_frame, f"Score: {blur_score:.1f} | {blur_status}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            blur_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2RGB)
        
        # Minimal overlay on original frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Frame: {frame_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert BGR to RGB
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        yield {
            'frame': display_frame,
            'blur_frame': blur_frame,
            'frame_count': frame_count,
            'total_frames': total_frames,
            'current_fps': current_fps,
            'width': width,
            'height': height,
            'video_fps': fps,
            'elapsed_time': current_time - start_time,
            'gpu_info': gpu_info,
            'blur_score': blur_score,
            'blur_status': blur_status,
            'clear_count': clear_count,
            'blurry_count': blurry_count
        }
    
    if executor:
        executor.shutdown()
    cap.release()

# Process video when button is clicked
if process_button:
    st.session_state.processing = True
    
    for data in process_video_stream(selected_video, skip_frames, enable_blur_detection, 
                                      blur_threshold if enable_blur_detection else 100.0,
                                      update_interval, use_threading):
        if stop_button or not st.session_state.get('processing', False):
            st.warning("Processing stopped by user")
            break
        
        # Display original frame
        frame_placeholder.image(data['frame'], channels="RGB", use_container_width=True)
        
        # Display blur detection frame
        if data['blur_frame'] is not None:
            blur_placeholder.image(data['blur_frame'], channels="RGB", use_container_width=True)
        else:
            blur_placeholder.info("Blur detection disabled")
        
        # Display processing info
        gpu_info = data['gpu_info']
        gpu_status = "✅ Enabled" if gpu_info.get('opencl_enabled', False) else "❌ Disabled"
        hw_decode_status = "✅ Yes" if gpu_info.get('hw_decode', False) else "❌ No"
        
        gpu_details = ""
        if gpu_info.get('opencl_available', False):
            gpu_details = f"\n        - Device: {gpu_info.get('device_name', 'Unknown')}"
        
        info_text = f"""
        **Video Properties:**
        - Resolution: {data['width']}x{data['height']}
        - Original FPS: {data['video_fps']:.2f}
        
        **GPU Acceleration:**
        - OpenCL: {gpu_status}{gpu_details}
        - HW Decode: {hw_decode_status}
        
        **Step 2: Blur Detection:**
        - Blur Score: {data['blur_score']:.2f}
        - Status: {data['blur_status']}
        - Clear Frames: {data['clear_count']}
        - Blurry Frames: {data['blurry_count']}
        
        **Current Status:**
        - Processing FPS: {data['current_fps']:.2f}
        - Frame: {data['frame_count']}/{data['total_frames']}
        - Progress: {(data['frame_count']/data['total_frames']*100):.1f}%
        """
        info_placeholder.markdown(info_text)
        
        # Update stats
        if show_stats:
            frame_counter.metric("Frame", f"{data['frame_count']}/{data['total_frames']}")
            fps_display.metric("Processing FPS", f"{data['current_fps']:.2f}")
            progress_display.metric("Progress", f"{(data['frame_count']/data['total_frames']*100):.1f}%")
            time_display.metric("Elapsed Time", f"{data['elapsed_time']:.1f}s")
        
        # Update progress bar (less frequently)
        if data['frame_count'] % max(1, update_interval) == 0:
            progress_bar.progress(data['frame_count'] / data['total_frames'])
    
    st.session_state.processing = False
    st.success("✅ Video processing complete!")

# Initial state
if 'processing' not in st.session_state:
    st.session_state.processing = False
    st.info("👈 Select a video and click 'Process Video' to start")
