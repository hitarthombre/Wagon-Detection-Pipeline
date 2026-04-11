import streamlit as st
import cv2
import time
import numpy as np
from pathlib import Path

st.set_page_config(page_title="AI Video Processing Pipeline", layout="wide")

st.title("🎥 AI Video Processing Pipeline")
st.markdown("### Step 1: Video Input Acquisition & Validation")

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

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📹 Original Frame")
    frame_placeholder = st.empty()

with col2:
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

def process_video_stream(video_path, skip_frames=0):
    """Process video and yield frames with metadata"""
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
    
    # Try hardware-accelerated video decoding (Intel Quick Sync, DXVA, etc.)
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_ANY)
    
    # Enable hardware acceleration backend if available
    try:
        # Try to use hardware acceleration for decoding
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
    start_time = time.time()
    prev_time = start_time
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames if requested
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            continue
        
        current_time = time.time()
        elapsed = current_time - prev_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        prev_time = current_time
        
        # Add overlay to frame using GPU-accelerated operations if available
        if gpu_info.get('opencl_enabled', False):
            # Use UMat for GPU operations
            gpu_frame = cv2.UMat(frame)
            cv2.putText(gpu_frame, f"Frame: {frame_count}/{total_frames}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(gpu_frame, f"FPS: {current_fps:.2f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(gpu_frame, "GPU", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            display_frame = gpu_frame.get()
        else:
            # CPU operations
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.2f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "CPU", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Convert BGR to RGB for Streamlit
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        yield {
            'frame': display_frame,
            'frame_count': frame_count,
            'total_frames': total_frames,
            'current_fps': current_fps,
            'width': width,
            'height': height,
            'video_fps': fps,
            'elapsed_time': current_time - start_time,
            'gpu_info': gpu_info
        }
    
    cap.release()

# Process video when button is clicked
if process_button:
    st.session_state.processing = True
    
    for data in process_video_stream(selected_video, skip_frames):
        if stop_button or not st.session_state.get('processing', False):
            st.warning("Processing stopped by user")
            break
        
        # Display frame
        frame_placeholder.image(data['frame'], channels="RGB", use_container_width=True)
        
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
        
        # Update progress bar
        progress_bar.progress(data['frame_count'] / data['total_frames'])
        
        # Small delay to prevent overwhelming the UI
        time.sleep(0.01)
    
    st.session_state.processing = False
    st.success("✅ Video processing complete!")

# Initial state
if 'processing' not in st.session_state:
    st.session_state.processing = False
    st.info("👈 Select a video and click 'Process Video' to start")
