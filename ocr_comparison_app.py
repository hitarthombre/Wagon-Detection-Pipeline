"""
Streamlit OCR Comparison Application
Compare TrOCR, PaddleOCR 2.7.x, and Tesseract OCR
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from ocr_engines import create_ocr_engine, TrOCREngine, PaddleOCREngine, TesseractEngine

# Page configuration
st.set_page_config(
    page_title="OCR Engine Comparison",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .engine-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🔍 OCR Engine Comparison Tool</h1>', unsafe_allow_html=True)
st.markdown("Compare **TrOCR**, **PaddleOCR 2.7.x**, and **Tesseract** OCR engines")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# OCR Engine Selection
engine_choice = st.sidebar.radio(
    "Select OCR Engine",
    options=["TrOCR (Transformer)", "PaddleOCR 2.7.x", "Tesseract OCR"],
    help="Choose one OCR engine to test"
)

# Engine-specific options
if engine_choice == "PaddleOCR 2.7.x":
    use_gpu = st.sidebar.checkbox("Use GPU", value=True, help="Enable GPU acceleration for PaddleOCR")
elif engine_choice == "Tesseract OCR":
    digits_only = st.sidebar.checkbox("Digits Only", value=False, help="Restrict to digits (0-9)")

st.sidebar.divider()

# Image source selection
image_source = st.sidebar.radio(
    "Image Source",
    options=["Upload Image", "Use Webcam", "Sample Images"],
    help="Choose how to provide the image"
)

st.sidebar.divider()

# Display options
show_preprocessing = st.sidebar.checkbox("Show Preprocessing", value=False)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)


# Cache OCR engines to avoid reloading
@st.cache_resource
def load_trocr_engine():
    """Load and cache TrOCR engine"""
    engine = TrOCREngine()
    if engine.initialize():
        return engine
    return None


@st.cache_resource
def load_paddleocr_engine(_use_gpu=True):
    """Load and cache PaddleOCR engine"""
    engine = PaddleOCREngine(use_gpu=_use_gpu)
    if engine.initialize():
        return engine
    return None


@st.cache_resource
def load_tesseract_engine(_digits_only=False):
    """Load and cache Tesseract engine"""
    engine = TesseractEngine(digits_only=_digits_only)
    if engine.initialize():
        return engine
    return None


def draw_boxes_on_image(image, boxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image"""
    img_with_boxes = image.copy()
    for box in boxes:
        points = np.array(box, dtype=np.int32)
        cv2.polylines(img_with_boxes, [points], True, color, thickness)
    return img_with_boxes


def process_image_with_ocr(image, engine):
    """Process image with selected OCR engine"""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    # Run OCR
    result = engine.extract_text(image_bgr)
    
    return result, image_bgr


# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Input Image")
    
    image = None
    
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload an image containing text"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    elif image_source == "Use Webcam":
        camera_image = st.camera_input("Take a picture")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            st.image(image, caption="Webcam Image", use_container_width=True)
    
    elif image_source == "Sample Images":
        sample_choice = st.selectbox(
            "Select Sample",
            options=["Text Document", "Receipt", "License Plate", "Handwritten Note"]
        )
        
        st.info("Sample images feature - upload your own images for now")

with col2:
    st.subheader("📝 OCR Results")
    
    if image is not None:
        # Load selected engine
        engine = None
        
        with st.spinner(f"Loading {engine_choice}..."):
            if engine_choice == "TrOCR (Transformer)":
                engine = load_trocr_engine()
            elif engine_choice == "PaddleOCR 2.7.x":
                engine = load_paddleocr_engine(use_gpu)
            elif engine_choice == "Tesseract OCR":
                digits_only_val = digits_only if 'digits_only' in locals() else False
                engine = load_tesseract_engine(digits_only_val)
        
        if engine is None:
            st.error(f"❌ Failed to load {engine_choice}. Please check installation.")
            st.info("Installation instructions:")
            if engine_choice == "TrOCR (Transformer)":
                st.code("pip install transformers torch pillow")
            elif engine_choice == "PaddleOCR 2.7.x":
                st.code("pip install paddleocr==2.7.0.3")
            elif engine_choice == "Tesseract OCR":
                st.code("pip install pytesseract")
                st.markdown("Also install Tesseract binary: [Download](https://github.com/UB-Mannheim/tesseract/wiki)")
        else:
            # Process button
            if st.button("🚀 Run OCR", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    result, image_bgr = process_image_with_ocr(image, engine)
                
                # Display results
                if result['success']:
                    st.markdown('<div class="success-box">✅ OCR Completed Successfully</div>', unsafe_allow_html=True)
                    
                    # Metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                    
                    with metric_col2:
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                    
                    with metric_col3:
                        text_length = len(result['text'])
                        st.metric("Characters", text_length)
                    
                    st.divider()
                    
                    # Extracted text
                    st.subheader("Extracted Text")
                    if result['text']:
                        st.text_area(
                            "Text Output",
                            value=result['text'],
                            height=150,
                            label_visibility="collapsed"
                        )
                    else:
                        st.warning("No text detected")
                    
                    # Detailed results
                    if 'details' in result and result['details']:
                        with st.expander("📊 Detailed Results"):
                            for i, (text, conf) in enumerate(result['details'], 1):
                                st.write(f"{i}. **{text}** (confidence: {conf:.2%})")
                    
                    # Bounding boxes visualization
                    if show_boxes and result['boxes']:
                        st.divider()
                        st.subheader("Bounding Boxes")
                        
                        # Draw boxes
                        image_with_boxes = draw_boxes_on_image(
                            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                            result['boxes']
                        )
                        
                        st.image(image_with_boxes, caption="Detected Text Regions", use_container_width=True)
                
                else:
                    st.markdown(f'<div class="error-box">❌ OCR Failed: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Please provide an image using the options in the left column")

# Information section
st.divider()

with st.expander("ℹ️ About OCR Engines"):
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("### TrOCR")
        st.markdown("""
        - **Type:** Transformer-based
        - **Strengths:** Handwritten text, complex layouts
        - **Speed:** Moderate
        - **Accuracy:** High for handwritten
        - **GPU:** Recommended
        """)
    
    with col_info2:
        st.markdown("### PaddleOCR 2.7.x")
        st.markdown("""
        - **Type:** Deep learning (CNN+RNN)
        - **Strengths:** Multi-language, fast
        - **Speed:** Fast
        - **Accuracy:** High for printed text
        - **GPU:** Optional
        """)
    
    with col_info3:
        st.markdown("### Tesseract")
        st.markdown("""
        - **Type:** Traditional OCR
        - **Strengths:** Printed text, digits
        - **Speed:** Very fast
        - **Accuracy:** Good with preprocessing
        - **GPU:** Not required
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | OCR Engine Comparison Tool</p>
    <p><small>Supports: TrOCR (HuggingFace) | PaddleOCR 2.7.x | Tesseract OCR</small></p>
</div>
""", unsafe_allow_html=True)
