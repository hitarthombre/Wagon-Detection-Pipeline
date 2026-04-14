import streamlit as st
import cv2
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import easyocr
from ocr_database import OCRDatabase

st.set_page_config(page_title="AI Video Processing Pipeline", layout="wide")

# Custom CSS for clean, compact layout
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    div[data-testid="column"] {
        background-color: #1e2130;
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #2e3140;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .main-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-align: center;
        color: #ffffff;
    }
    .stImage {
        border-radius: 6px;
        overflow: hidden;
    }
    h3 {
        margin-top: 0 !important;
        padding-top: 0 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    .ocr-data-box {
        background-color: #1a1d2e;
        border: 1px solid #2e3140;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .ocr-entry {
        background-color: #252837;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
        border-left: 3px solid #10b981;
    }
    .ocr-timestamp {
        color: #10b981;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .ocr-text {
        color: #ffffff;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .ocr-confidence {
        color: #a0a0a0;
        font-size: 0.8rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🎥 AI Video Processing Pipeline</h1>', unsafe_allow_html=True)

# Initialize OCR Database
@st.cache_resource
def get_ocr_database():
    return OCRDatabase()

ocr_db = get_ocr_database()

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Tab for Process, Logs, and Results
    tab1, tab2, tab3 = st.tabs(["▶️ Process", "📋 Logs", "🖼️ Results"])
    
    with tab1:
        # Input source selection
        input_source = st.radio(
            "Input Source",
            ["📁 Video File", "📷 Camera"],
            horizontal=True
        )
        
        selected_video = None
        use_camera = False
        camera_index = 0
        
        if input_source == "📁 Video File":
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
        else:
            use_camera = True
            camera_index = st.number_input(
                "Camera Index",
                min_value=0,
                max_value=10,
                value=0,
                help="Usually 0 for built-in webcam, 1+ for external cameras"
            )
            st.info("📷 Live camera feed will be processed in real-time")
        
        process_button = st.button("▶️ Start Processing", type="primary", width="stretch")
        stop_button = st.button("⏹️ Stop", width="stretch")
    
    with tab2:
        st.subheader("📊 Previous Scans")
        all_videos = ocr_db.get_all_videos()
        
        if all_videos:
            st.metric("Total Scans", len(all_videos))
            
            for idx, video in enumerate(reversed(all_videos[-10:])):  # Show last 10
                with st.expander(f"📹 {video['video_name']}", expanded=False):
                    st.write(f"**Scan ID:** {video['video_id']}")
                    st.write(f"**Date:** {video['timestamp'][:19]}")
                    st.write(f"**Frames:** {video['total_frames_processed']}")
                    st.write(f"**Text Detected:** {video['total_text_detected']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**With Numbers:** {len(video['text_with_numbers'])}")
                    with col2:
                        st.write(f"**Text Only:** {len(video['text_only'])}")
                    
                    if st.button(f"📄 Export Report", key=f"export_{idx}"):
                        report_path = ocr_db.export_video_report(video['video_id'])
                        st.success(f"Report saved: {report_path}")
        else:
            st.info("No previous scans found. Process a video to see logs here.")
    
    with tab3:
        st.subheader("🖼️ View Results")
        all_videos = ocr_db.get_all_videos()
        
        if all_videos:
            # Select video to view
            video_names = [f"{v['video_name']} ({v['timestamp'][:19]})" for v in reversed(all_videos)]
            selected_result_idx = st.selectbox(
                "Select Scan",
                range(len(video_names)),
                format_func=lambda x: video_names[x]
            )
            
            selected_result = list(reversed(all_videos))[selected_result_idx]
            
            # Export buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Export TXT", width="stretch"):
                    report_path = ocr_db.export_video_report(selected_result['video_id'])
                    st.success(f"✅ Saved: {report_path.name}")
            
            with col2:
                if st.button("📕 Export PDF", width="stretch"):
                    with st.spinner("Generating PDF..."):
                        pdf_path = ocr_db.export_video_pdf(selected_result['video_id'])
                        if pdf_path:
                            st.success(f"✅ Saved: {pdf_path.name}")
                        else:
                            st.error("Install reportlab: pip install reportlab")
        else:
            st.info("No results yet. Process a video with OCR enabled.")
    
    st.divider()
    st.subheader("⚡ Performance")
    skip_frames = st.slider("Skip Frames", 0, 10, 0)
    update_interval = st.slider("UI Update Interval", 1, 30, 5, 
                                help="Update UI every N frames for better performance")
    
    st.divider()
    st.subheader("🔧 Pipeline Steps")
    enable_blur_detection = st.checkbox("Step 2: Blur Detection", value=True)
    if enable_blur_detection:
        blur_threshold = st.slider("Blur Threshold", 50.0, 300.0, 100.0, 10.0)
    
    enable_enhancement = st.checkbox("Step 3: Frame Enhancement", value=True,
                                    help="Enhance only blurry frames")
    if enable_enhancement:
        enhancement_strength = st.slider("Enhancement Strength", 1.0, 2.0, 1.1, 0.1)
    
    enable_ocr = st.checkbox("Step 4: OCR Text Extraction", value=False,
                            help="Extract text from frames using EasyOCR")

    
    if enable_ocr:
        use_batch_ocr = st.checkbox("Enable Batch Processing (GPU)", value=True,
                                   help="Process multiple frames at once for better GPU utilization")
        if use_batch_ocr:
            batch_size = st.slider("Batch Size", 2, 8, 4,
                                  help="Number of frames to process together (higher = faster but more VRAM)")
        
        ocr_confidence_threshold = st.slider("OCR Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
        ocr_interval = st.slider("OCR Capture Interval (seconds)", 1, 10, 1,
                                help="Capture OCR data every N seconds")
    
    st.divider()
    st.subheader("💾 Video Recording")
    save_video = st.checkbox("Save Processed Video", value=False,
                            help="Save all 4 stages as video file")
    if save_video:
        output_filename = st.text_input("Output Filename", "processed_output.mp4")

# Results Viewer Section (when viewing previous results)
if 'selected_result' in locals() and selected_result and not process_button:
    st.markdown("## 🖼️ Viewing Previous Results")
    st.markdown(f"**Video:** {selected_result['video_name']} | **Date:** {selected_result['timestamp'][:19]}")
    
    # Summary metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Total Frames", selected_result['total_frames_processed'])
    with metric_col2:
        st.metric("Text Detected", selected_result['total_text_detected'])
    with metric_col3:
        st.metric("With Numbers", len(selected_result['text_with_numbers']))
    with metric_col4:
        st.metric("Text Only", len(selected_result['text_only']))
    
    st.divider()
    
    # Two columns for categorized text
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 🔢 Text with Numbers")
        if selected_result['text_with_numbers']:
            for item in selected_result['text_with_numbers']:
                st.info(f"**{item['text']}**  \nConfidence: {item['confidence']:.2%} | Frame: {item['first_seen_frame']}")
        else:
            st.write("No text with numbers detected")
    
    with col_right:
        st.markdown("### 📝 Text Only")
        if selected_result['text_only']:
            for item in selected_result['text_only']:
                st.success(f"**{item['text']}**  \nConfidence: {item['confidence']:.2%} | Frame: {item['first_seen_frame']}")
        else:
            st.write("No text-only detected")
    
    st.divider()
    
    # Display frames with detected text
    st.markdown("### 🖼️ Frames with Detected Text")
    
    if selected_result['frames_with_text']:
        # Display in grid
        cols_per_row = 2
        frames_data = selected_result['frames_with_text']
        
        for i in range(0, len(frames_data), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(frames_data):
                    frame_data = frames_data[i + j]
                    frame_path = ocr_db.get_frame_image_path(frame_data['frame_image'])
                    
                    with col:
                        if frame_path.exists():
                            st.image(str(frame_path), caption=f"Frame {frame_data['frame_number']}", use_column_width=True)
                            texts = ", ".join([f"{t['text']} ({t['confidence']:.2%})" for t in frame_data['texts']])
                            st.caption(f"📝 {texts}")
                        else:
                            st.warning(f"Frame {frame_data['frame_number']} image not found")
    else:
        st.info("No frames with detected text")
    
    st.stop()  # Don't show processing interface when viewing results

# Main content area - compact layout
col1, col2, col3, col4 = st.columns([3, 3, 3, 3], gap="small")

with col1:
    st.markdown("### 📹 Original")
    frame_placeholder = st.empty()

with col2:
    st.markdown("### 🔍 Blur Detection")
    blur_placeholder = st.empty()

with col3:
    st.markdown("### ✨ Enhanced")
    enhance_placeholder = st.empty()

with col4:
    st.markdown("### 📝 OCR")
    ocr_placeholder = st.empty()

# Stats area - compact
st.divider()
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4, gap="small")
with stats_col1:
    frame_counter = st.empty()
with stats_col2:
    fps_display = st.empty()
with stats_col3:
    progress_display = st.empty()
with stats_col4:
    time_display = st.empty()

progress_bar = st.progress(0)

# OCR Data Display Section
if enable_ocr:
    st.divider()
    st.markdown("### 📊 OCR Results")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🔢 Text with Numbers")
        text_with_numbers_container = st.empty()
    
    with col_right:
        st.markdown("#### 📝 Text Only")
        text_only_container = st.empty()
    
    st.divider()
    st.markdown("### 🖼️ Frames with Detected Text")
    frames_container = st.empty()


def get_image_base64(image_path):
    """Convert image to base64 for HTML display"""
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def detect_blur(frame, threshold=100.0):
    """Detect blur using Laplacian variance method - GPU optimized"""
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
    """Enhanced preprocessing for better OCR accuracy"""
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

def perform_ocr(frame, ocr_engine, confidence_threshold=0.7, use_preprocessing=True):
    """Perform OCR on frame using EasyOCR with enhanced preprocessing"""
    ocr_frame = frame.copy()
    text_results = []
    
    try:
        # EasyOCR with preprocessing
        if use_preprocessing:
            processed_frame = preprocess_for_ocr(frame)
            results_original = ocr_engine.readtext(frame)
            results_processed = ocr_engine.readtext(processed_frame)
            
            combined_results = {}
            for bbox, text, confidence in results_original + results_processed:
                if text not in combined_results or confidence > combined_results[text][1]:
                    combined_results[text] = (bbox, confidence)
            
            results = [(bbox, text, conf) for text, (bbox, conf) in combined_results.items()]
        else:
            results = ocr_engine.readtext(frame)
        
        for (bbox, text, confidence) in results:
            if confidence >= confidence_threshold:
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(ocr_frame, [pts], True, (0, 255, 0), 2)
                
                label = f"{text} ({confidence:.2f})"
                text_pos = (int(pts[0][0]), int(pts[0][1]) - 10)
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                cv2.rectangle(
                    ocr_frame,
                    (text_pos[0], text_pos[1] - text_height - 5),
                    (text_pos[0] + text_width, text_pos[1] + 5),
                    (0, 255, 0), -1
                )
                
                cv2.putText(ocr_frame, label, text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                text_results.append({
                    'text': text, 
                    'confidence': confidence
                })
    
    except Exception as e:
        print(f"OCR error: {e}")
    
    return ocr_frame, text_results


def perform_ocr_batch(frames, ocr_engine, confidence_threshold=0.7, use_preprocessing=True):
    """Perform OCR on multiple frames at once using GPU batch processing"""
    ocr_frames = []
    all_text_results = []
    
    try:
        # Prepare frames for batch processing
        frame_list = []
        processed_list = []
        
        for frame in frames:
            frame_list.append(frame)
            if use_preprocessing:
                processed_list.append(preprocess_for_ocr(frame))
        
        # Batch process all frames at once on GPU
        if use_preprocessing and processed_list:
            # Process both original and preprocessed in batches
            results_original = ocr_engine.readtext_batched(frame_list, n_width=320, n_height=64)
            results_processed = ocr_engine.readtext_batched(processed_list, n_width=320, n_height=64)
            
            # Combine results for each frame
            batch_results = []
            for i in range(len(frames)):
                combined_results = {}
                for bbox, text, confidence in results_original[i] + results_processed[i]:
                    if text not in combined_results or confidence > combined_results[text][1]:
                        combined_results[text] = (bbox, confidence)
                batch_results.append([(bbox, text, conf) for text, (bbox, conf) in combined_results.items()])
        else:
            # Process original frames only
            batch_results = ocr_engine.readtext_batched(frame_list, n_width=320, n_height=64)
        
        # Draw results on each frame
        for frame_idx, (frame, results) in enumerate(zip(frames, batch_results)):
            ocr_frame = frame.copy()
            text_results = []
            
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold:
                    pts = np.array(bbox, dtype=np.int32)
                    cv2.polylines(ocr_frame, [pts], True, (0, 255, 0), 2)
                    
                    label = f"{text} ({confidence:.2f})"
                    text_pos = (int(pts[0][0]), int(pts[0][1]) - 10)
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    
                    cv2.rectangle(
                        ocr_frame,
                        (text_pos[0], text_pos[1] - text_height - 5),
                        (text_pos[0] + text_width, text_pos[1] + 5),
                        (0, 255, 0), -1
                    )
                    
                    cv2.putText(ocr_frame, label, text_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    text_results.append({
                        'text': text, 
                        'confidence': confidence
                    })
            
            ocr_frames.append(ocr_frame)
            all_text_results.append(text_results)
    
    except Exception as e:
        print(f"Batch OCR error: {e}, falling back to single frame processing")
        # Fallback to single frame processing
        for frame in frames:
            ocr_frame, text_results = perform_ocr(frame, ocr_engine, confidence_threshold, use_preprocessing)
            ocr_frames.append(ocr_frame)
            all_text_results.append(text_results)
    
    return ocr_frames, all_text_results


@st.cache_resource
def initialize_ocr_engine():
    """Initialize and cache EasyOCR engine"""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)
        return reader
    except Exception as e:
        st.error(f"Failed to initialize EasyOCR: {e}")
        return None


def process_video_stream(video_path, skip_frames=0, enable_blur=False, blur_threshold=100.0, 
                        update_interval=5, enable_enhance=False, enhance_strength=1.1,
                        enable_ocr=False, use_batch_ocr=False, batch_size=4, ocr_confidence=0.7, ocr_interval=1, 
                        save_video=False, output_filename="processed_output.mp4", ocr_db=None, video_id=None,
                        use_camera=False, camera_index=0):
    """Process video or camera feed with optimized performance and optional batch OCR"""
    # Initialize OCR if enabled
    ocr_engine = None
    ocr_frame_buffer = []  # Buffer for batch processing
    
    if enable_ocr:
        ocr_engine = initialize_ocr_engine()
        
        if ocr_engine is None:
            st.error("Failed to initialize EasyOCR")
            enable_ocr = False
    
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
    
    # Open video source (file or camera)
    if use_camera:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for live feed
        source_name = f"Camera_{camera_index}"
    else:
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        source_name = video_path.name
        
        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            gpu_info['hw_decode'] = True
        except:
            gpu_info['hw_decode'] = False
    
    if not cap.isOpened():
        st.error(f"Cannot open {'camera' if use_camera else 'video'}: {camera_index if use_camera else video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) if not use_camera else 30.0  # Default 30 FPS for camera
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not use_camera else 999999  # Infinite for camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer if saving
    video_writer = None
    if save_video and not use_camera:  # Don't save camera feed by default
        # Create 2x2 grid size (each frame will be resized to fit)
        grid_width = 1280
        grid_height = 960
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (grid_width, grid_height))
        st.info(f"📹 Recording to: {output_filename} ({grid_width}x{grid_height})")
    
    frame_count = 0
    blurry_count = 0
    clear_count = 0
    enhanced_count = 0
    skipped_count = 0
    total_text_detected = 0
    start_time = time.time()
    prev_time = start_time
    last_ocr_time = start_time
    ocr_data_log = []  # Store OCR data with timestamps
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            continue
        
        if frame_count % update_interval != 0:
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
            
            blur_frame = frame.copy()
            cv2.putText(blur_frame, f"Score: {blur_score:.1f} | {blur_status}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            blur_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2RGB)
        
        enhanced_frame = None
        was_enhanced = False
        enhanced_result = frame
        
        if enable_enhance and enable_blur:
            enhanced_result, was_enhanced = enhance_frame(frame, blur_status, enhance_strength)
            if was_enhanced:
                enhanced_count += 1
                enhance_color = (0, 165, 255)
                enhance_text = "ENHANCED"
            else:
                skipped_count += 1
                enhance_color = (0, 255, 0)
                enhance_text = "ORIGINAL"
            
            enhanced_frame = enhanced_result.copy()
            cv2.putText(enhanced_frame, f"{enhance_text}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, enhance_color, 2)
            enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        
        ocr_frame = None
        text_results = []
        should_capture_ocr = False
        
        if enable_ocr and ocr_engine:
            # Check if it's time to capture OCR (every N seconds)
            if current_time - last_ocr_time >= ocr_interval:
                should_capture_ocr = True
                last_ocr_time = current_time
            
            # Batch processing mode
            if use_batch_ocr:
                # Add frame to buffer
                ocr_frame_buffer.append((enhanced_result.copy(), frame_count, should_capture_ocr))
                
                # Process batch when buffer is full
                if len(ocr_frame_buffer) >= batch_size:
                    frames_to_process = [item[0] for item in ocr_frame_buffer]
                    frame_numbers = [item[1] for item in ocr_frame_buffer]
                    capture_flags = [item[2] for item in ocr_frame_buffer]
                    
                    # Batch process all frames at once
                    ocr_results, batch_text_results = perform_ocr_batch(
                        frames_to_process, ocr_engine, ocr_confidence
                    )
                    
                    # Store results for each frame
                    for idx, (ocr_res, texts, frame_num, should_capture) in enumerate(
                        zip(ocr_results, batch_text_results, frame_numbers, capture_flags)
                    ):
                        total_text_detected += len(texts)
                        
                        # Log OCR data if it's time to capture
                        if should_capture and texts:
                            elapsed_seconds = int(current_time - start_time)
                            ocr_data_log.append({
                                'timestamp': elapsed_seconds,
                                'frame': frame_num,
                                'texts': texts
                            })
                            
                            # Save to database with frame image
                            if ocr_db and video_id:
                                ocr_db.add_ocr_result(video_id, frame_num, enhanced_result, texts)
                        
                        # Only keep the last result for display
                        if idx == len(ocr_results) - 1:
                            ocr_result = ocr_res
                            text_results = texts
                    
                    # Clear buffer
                    ocr_frame_buffer.clear()
                else:
                    # Use previous result or blank frame while buffering
                    ocr_result = enhanced_result.copy()
                    text_results = []
            else:
                # Single frame processing (original behavior)
                ocr_result, text_results = perform_ocr(enhanced_result, ocr_engine, ocr_confidence)
                total_text_detected += len(text_results)
                
                # Log OCR data if it's time to capture
                if should_capture_ocr and text_results:
                    elapsed_seconds = int(current_time - start_time)
                    ocr_data_log.append({
                        'timestamp': elapsed_seconds,
                        'frame': frame_count,
                        'texts': text_results
                    })
                    
                    # Save to database with frame image
                    if ocr_db and video_id:
                        ocr_db.add_ocr_result(video_id, frame_count, enhanced_result, text_results)
            
            if 'ocr_result' in locals():
                ocr_frame = ocr_result.copy()
                cv2.putText(ocr_frame, f"Text: {len(text_results)}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if use_batch_ocr:
                    cv2.putText(ocr_frame, f"Batch: {batch_size}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                ocr_frame = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2RGB)
        
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Frame: {frame_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Save to video if enabled
        if video_writer is not None:
            # Create grid for video output
            frame_size = (640, 480)
            
            # Resize all frames
            orig_resized = cv2.resize(frame, frame_size)
            blur_resized = cv2.resize(cv2.cvtColor(blur_frame, cv2.COLOR_RGB2BGR) if blur_frame is not None else frame, frame_size)
            enhance_resized = cv2.resize(cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR) if enhanced_frame is not None else frame, frame_size)
            ocr_resized = cv2.resize(cv2.cvtColor(ocr_frame, cv2.COLOR_RGB2BGR) if ocr_frame is not None else frame, frame_size)
            
            # Add labels
            def add_label(img, text):
                labeled = img.copy()
                cv2.rectangle(labeled, (0, 0), (640, 40), (0, 0, 0), -1)
                cv2.putText(labeled, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return labeled
            
            orig_labeled = add_label(orig_resized, "Step 1: Original")
            blur_labeled = add_label(blur_resized, "Step 2: Blur Detection")
            enhance_labeled = add_label(enhance_resized, "Step 3: Enhanced")
            ocr_labeled = add_label(ocr_resized, "Step 4: OCR")
            
            # Create 2x2 grid
            top_row = np.hstack([orig_labeled, blur_labeled])
            bottom_row = np.hstack([enhance_labeled, ocr_labeled])
            grid = np.vstack([top_row, bottom_row])
            
            # Write frame
            video_writer.write(grid)
        
        yield {
            'frame': display_frame,
            'blur_frame': blur_frame,
            'enhanced_frame': enhanced_frame,
            'ocr_frame': ocr_frame,
            'text_results': text_results,
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
            'blurry_count': blurry_count,
            'enhanced_count': enhanced_count,
            'skipped_count': skipped_count,
            'total_text_detected': total_text_detected,
            'ocr_data_log': ocr_data_log
        }
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()

# Process video when button is clicked
if process_button:
    st.session_state.processing = True
    
    # Create new video log entry
    source_name = f"Camera_{camera_index}" if use_camera else selected_video.name
    video_id = ocr_db.create_video_log(source_name) if enable_ocr else None
    
    for data in process_video_stream(
        selected_video if not use_camera else None,
        skip_frames, enable_blur_detection, 
        blur_threshold if enable_blur_detection else 100.0,
        update_interval, enable_enhancement,
        enhancement_strength if enable_enhancement else 1.1,
        enable_ocr,
        use_batch_ocr if enable_ocr else False,
        batch_size if (enable_ocr and use_batch_ocr) else 4,
        ocr_confidence_threshold if enable_ocr else 0.7,
        ocr_interval if enable_ocr else 1,
        save_video and not use_camera,  # Don't save camera feed
        output_filename if save_video else "processed_output.mp4",
        ocr_db if enable_ocr else None,
        video_id if enable_ocr else None,
        use_camera,
        camera_index if use_camera else 0
    ):
        if stop_button or not st.session_state.get('processing', False):
            st.warning("Processing stopped by user")
            break
        
        frame_placeholder.image(data['frame'], channels="RGB", width="stretch")
        
        if data['blur_frame'] is not None:
            blur_placeholder.image(data['blur_frame'], channels="RGB", width="stretch")
        else:
            blur_placeholder.info("Disabled")
        
        if data['enhanced_frame'] is not None:
            enhance_placeholder.image(data['enhanced_frame'], channels="RGB", width="stretch")
        else:
            enhance_placeholder.info("Disabled")
        
        if data['ocr_frame'] is not None:
            ocr_placeholder.image(data['ocr_frame'], channels="RGB", width="stretch")
        else:
            ocr_placeholder.info("Disabled")
        
        # Update stats
        frame_counter.metric("Frame", f"{data['frame_count']}/{data['total_frames']}")
        fps_display.metric("FPS", f"{data['current_fps']:.1f}")
        progress_display.metric("Progress", f"{(data['frame_count']/data['total_frames']*100):.1f}%")
        time_display.metric("Time", f"{data['elapsed_time']:.1f}s")
        
        progress_bar.progress(data['frame_count'] / data['total_frames'])
        
        # Display OCR results from database
        if enable_ocr and video_id:
            video_data = ocr_db.get_video_by_id(video_id)
            if video_data:
                # Text with numbers column
                if video_data['text_with_numbers']:
                    numbers_html = '<div style="max-height: 400px; overflow-y: auto;">'
                    for item in video_data['text_with_numbers']:
                        numbers_html += f'<div style="padding: 8px; margin: 4px 0; background: #2e3140; border-radius: 4px; border-left: 3px solid #10b981;">'
                        numbers_html += f'<div style="color: #10b981; font-weight: 600;">{item["text"]}</div>'
                        numbers_html += f'<div style="color: #a0a0a0; font-size: 0.85em;">Confidence: {item["confidence"]:.2%} | Frame: {item["first_seen_frame"]}</div>'
                        numbers_html += '</div>'
                    numbers_html += '</div>'
                    text_with_numbers_container.markdown(numbers_html, unsafe_allow_html=True)
                else:
                    text_with_numbers_container.info("No text with numbers detected yet")
                
                # Text only column
                if video_data['text_only']:
                    text_html = '<div style="max-height: 400px; overflow-y: auto;">'
                    for item in video_data['text_only']:
                        text_html += f'<div style="padding: 8px; margin: 4px 0; background: #2e3140; border-radius: 4px; border-left: 3px solid #3b82f6;">'
                        text_html += f'<div style="color: #3b82f6; font-weight: 600;">{item["text"]}</div>'
                        text_html += f'<div style="color: #a0a0a0; font-size: 0.85em;">Confidence: {item["confidence"]:.2%} | Frame: {item["first_seen_frame"]}</div>'
                        text_html += '</div>'
                    text_html += '</div>'
                    text_only_container.markdown(text_html, unsafe_allow_html=True)
                else:
                    text_only_container.info("No text-only detected yet")
                
                # Display frames with text
                if video_data['frames_with_text']:
                    frames_html = '<div style="max-height: 300px; overflow-y: auto;">'
                    for frame_data in reversed(video_data['frames_with_text'][-5:]):  # Show last 5 frames
                        frame_path = ocr_db.get_frame_image_path(frame_data['frame_image'])
                        if frame_path.exists():
                            frames_html += f'<div style="margin: 10px 0; padding: 10px; background: #1e2130; border-radius: 8px;">'
                            frames_html += f'<div style="color: #10b981; font-weight: 600; margin-bottom: 5px;">Frame {frame_data["frame_number"]}</div>'
                            frames_html += f'<img src="data:image/jpeg;base64,{get_image_base64(str(frame_path))}" style="width: 100%; border-radius: 4px; margin: 5px 0;">'
                            frames_html += '<div style="margin-top: 5px;">'
                            for text in frame_data['texts']:
                                frames_html += f'<span style="background: #10b981; color: black; padding: 2px 6px; border-radius: 3px; margin: 2px; display: inline-block; font-size: 0.85em;">{text["text"]}</span>'
                            frames_html += '</div></div>'
                    frames_html += '</div>'
                    frames_container.markdown(frames_html, unsafe_allow_html=True)
    
    st.session_state.processing = False
    st.success("✅ Video processing complete!")
    
    if enable_ocr and video_id:
        report_path = ocr_db.export_video_report(video_id)
        st.success(f"📄 OCR Report saved: {report_path}")
    
    if save_video:
        st.success(f"💾 Video saved to: {output_filename if save_video else 'processed_output.mp4'}")
        st.info("You can download the file from your project directory")


if 'processing' not in st.session_state:
    st.session_state.processing = False
    st.info("👈 Select a video and click 'Process Video' to start")
