"""
Modular OCR Engine Classes
Supports: TrOCR, PaddleOCR 2.7.x, Tesseract
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import time


class OCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    @abstractmethod
    def initialize(self):
        """Initialize the OCR engine"""
        pass
    
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> Dict:
        """
        Extract text from image
        
        Returns:
            Dict with keys: 'text', 'confidence', 'boxes', 'processing_time'
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return engine name"""
        pass


class TrOCREngine(OCREngine):
    """TrOCR - Transformer-based OCR using HuggingFace"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = None
        
    def initialize(self):
        """Initialize TrOCR model"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            
            print("Loading TrOCR model...")
            # Use handwritten model for better general performance
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            
            # Use GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            print(f"TrOCR initialized on {self.device}")
            return True
        except Exception as e:
            print(f"Failed to initialize TrOCR: {e}")
            return False
    
    def extract_text(self, image: np.ndarray) -> Dict:
        """Extract text using TrOCR"""
        start_time = time.time()
        
        try:
            from PIL import Image
            import torch
            
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Process image
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            processing_time = time.time() - start_time
            
            return {
                'text': generated_text,
                'confidence': 0.95,  # TrOCR doesn't provide confidence scores
                'boxes': [],  # TrOCR doesn't provide bounding boxes
                'processing_time': processing_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'boxes': [],
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def get_name(self) -> str:
        return "TrOCR (Transformer)"


class PaddleOCREngine(OCREngine):
    """PaddleOCR 2.7.x - Stable version"""
    
    def __init__(self, use_gpu: bool = True):
        self.ocr = None
        self.use_gpu = use_gpu
        
    def initialize(self):
        """Initialize PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            import logging
            import os
            
            # Suppress PaddleOCR logging
            logging.getLogger('ppocr').setLevel(logging.ERROR)
            
            # Bypass model source check for faster startup
            os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
            
            print("Loading PaddleOCR...")
            # Initialize with minimal parameters (newer versions have different API)
            try:
                # Try with use_gpu parameter first (older versions)
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=self.use_gpu
                )
            except Exception as e:
                # If use_gpu not supported, try without it (newer versions)
                if "use_gpu" in str(e) or "Unknown argument" in str(e):
                    print("Note: use_gpu parameter not supported, using default device")
                    self.ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang='en'
                    )
                else:
                    raise
            print("PaddleOCR initialized")
            return True
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {e}")
            print("Try: pip install paddleocr")
            return False
            print("Try: pip install paddleocr")
            return False
    
    def extract_text(self, image: np.ndarray) -> Dict:
        """Extract text using PaddleOCR 2.7.x"""
        start_time = time.time()
        
        try:
            # Run OCR using .ocr() method (2.7.x API)
            result = self.ocr.ocr(image, cls=True)
            
            processing_time = time.time() - start_time
            
            # Parse results
            texts = []
            confidences = []
            boxes = []
            
            if result and result[0]:
                for line in result[0]:
                    box, (text, confidence) = line
                    texts.append(text)
                    confidences.append(confidence)
                    boxes.append(box)
            
            # Combine all text
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': full_text,
                'confidence': float(avg_confidence),
                'boxes': boxes,
                'processing_time': processing_time,
                'success': True,
                'details': list(zip(texts, confidences))
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'boxes': [],
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def get_name(self) -> str:
        return "PaddleOCR 2.7.x"


class TesseractEngine(OCREngine):
    """Tesseract OCR with preprocessing"""
    
    def __init__(self, digits_only: bool = False):
        self.digits_only = digits_only
        
    def initialize(self):
        """Initialize Tesseract"""
        try:
            import pytesseract
            import os
            
            # Try to set Tesseract path for Windows
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Tesseract-OCR\tesseract.exe'
            ]
            
            # Check if tesseract is in PATH or set explicit path
            tesseract_found = False
            try:
                pytesseract.get_tesseract_version()
                tesseract_found = True
            except:
                # Try explicit paths
                for path in possible_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        try:
                            pytesseract.get_tesseract_version()
                            tesseract_found = True
                            break
                        except:
                            continue
            
            if not tesseract_found:
                raise Exception("Tesseract executable not found")
            
            # Test if Tesseract is installed
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {version}")
            return True
        except Exception as e:
            print(f"Failed to initialize Tesseract: {e}")
            print("Make sure Tesseract is installed:")
            print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("  Linux: sudo apt-get install tesseract-ocr")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def extract_text(self, image: np.ndarray) -> Dict:
        """Extract text using Tesseract"""
        start_time = time.time()
        
        try:
            import pytesseract
            
            # Preprocess image
            processed = self.preprocess_image(image)
            
            # Configure Tesseract
            config = '--psm 6'  # Assume uniform block of text
            if self.digits_only:
                config += ' -c tessedit_char_whitelist=0123456789'
            
            # Extract text with confidence
            data = pytesseract.image_to_data(
                processed, 
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            processing_time = time.time() - start_time
            
            # Parse results
            texts = []
            confidences = []
            boxes = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 0:  # Filter out low confidence
                    text = data['text'][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(int(conf) / 100.0)  # Convert to 0-1 range
                        
                        # Get bounding box
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        boxes.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            
            # Combine all text
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': full_text,
                'confidence': float(avg_confidence),
                'boxes': boxes,
                'processing_time': processing_time,
                'success': True,
                'details': list(zip(texts, confidences))
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'boxes': [],
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def get_name(self) -> str:
        suffix = " (Digits Only)" if self.digits_only else ""
        return f"Tesseract{suffix}"


# Factory function to create OCR engines
def create_ocr_engine(engine_type: str, **kwargs) -> OCREngine:
    """
    Factory function to create OCR engine instances
    
    Args:
        engine_type: 'trocr', 'paddleocr', or 'tesseract'
        **kwargs: Additional arguments for specific engines
    
    Returns:
        OCREngine instance
    """
    engines = {
        'trocr': TrOCREngine,
        'paddleocr': PaddleOCREngine,
        'tesseract': TesseractEngine
    }
    
    if engine_type.lower() not in engines:
        raise ValueError(f"Unknown engine type: {engine_type}")
    
    return engines[engine_type.lower()](**kwargs)
