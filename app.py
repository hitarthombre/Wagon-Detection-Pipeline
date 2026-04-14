import streamlit as st
import cv2
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import easyocr

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

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
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
    
    process_button = st.button("▶️ Process Video", type="primary", width="stretch")
    stop_button = st.button("⏹️ Stop", width="stretch")
    
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
        ocr_confidence_threshold = st.slider("OCR Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
        ocr_interval = st.slider("OCR Capture Interval (seconds)", 1, 10, 1,
                                help="Capture OCR data every N seconds")
    
    st.divider()
    st.subheader("💾 Video Recording")
    save_video = st.checkbox("Save Processed Video", value=False,
                            help="Save all 4 stages as video file")
    if save_video:
        output_filename = st.text_input("Output Filename", "processed_output.mp4")

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
    st.markdown("### 📊 OCR Data (Captured Every Second)")
    ocr_data_container = st.empty()

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
        
        for (bbox, text, confidence) in results:
            if confidence >= confidence_threshold:
                # Draw bounding box
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(ocr_frame, [pts], True, (0, 255, 0), 2)
                
                # Prepare text with confidence
                label = f"{text} ({confidence:.2f})"
                
                # Calculate text size for background
                text_pos = (int(pts[0][0]), int(pts[0][1]) - 10)
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
                
                # Add text result
                text_results.append({
                    'text': text, 
                    'confidence': confidence
                })
    
    except Exception as e:
        print(f"OCR error: {e}")
    
    return ocr_frame, text_results


@st.cache_resource
def initialize_ocr_engine():
    """Initialize and cache EasyOCR engine"""
    try:
        reader = easyocr.Reader(['en'], gpu=True)
        return reader
    except Exception as e:
        print(f"Failed to initialize EasyOCR: {e}")
        return None


def process_video_stream(video_path, skip_frames=0, enable_blur=False, blur_threshold=100.0, 
                        update_interval=5, enable_enhance=False, enhance_strength=1.1,
                        enable_ocr=False, ocr_confidence=0.7, ocr_interval=1, 
                        save_video=False, output_filename="processed_output.mp4"):
    """Process video with optimized performance"""
    # Initialize OCR if enabled
    ocr_engine = None
    
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
    
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    try:
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        gpu_info['hw_decode'] = True
    except:
        gpu_info['hw_decode'] = False
    
    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer if saving
    video_writer = None
    if save_video:
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
            
            ocr_frame = ocr_result.copy()
            cv2.putText(ocr_frame, f"Text: {len(text_results)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
    
    for data in process_video_stream(
        selected_video, skip_frames, enable_blur_detection, 
        blur_threshold if enable_blur_detection else 100.0,
        update_interval, enable_enhancement,
        enhancement_strength if enable_enhancement else 1.1,
        enable_ocr, 
        ocr_confidence_threshold if enable_ocr else 0.7,
        ocr_interval if enable_ocr else 1,
        save_video, 
        output_filename if save_video else "processed_output.mp4"
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
        
        # Display OCR data log
        if enable_ocr and data['ocr_data_log']:
            ocr_html = '<div class="ocr-data-box">'
            for entry in reversed(data['ocr_data_log'][-10:]):  # Show last 10 entries
                ocr_html += f'<div class="ocr-entry">'
                ocr_html += f'<div class="ocr-timestamp">⏱️ {entry["timestamp"]}s (Frame {entry["frame"]})</div>'
                for text_item in entry['texts']:
                    ocr_html += f'<div class="ocr-text">📝 {text_item["text"]}</div>'
                    ocr_html += f'<div class="ocr-confidence">Confidence: {text_item["confidence"]:.2%}</div>'
                ocr_html += '</div>'
            ocr_html += '</div>'
            ocr_data_container.markdown(ocr_html, unsafe_allow_html=True)
    
    st.session_state.processing = False
    st.success("✅ Video processing complete!")
    
    if save_video:
        st.success(f"💾 Video saved to: {output_filename if save_video else 'processed_output.mp4'}")
        st.info("You can download the file from your project directory")

if 'processing' not in st.session_state:
    st.session_state.processing = False
    st.info("👈 Select a video and click 'Process Video' to start")
