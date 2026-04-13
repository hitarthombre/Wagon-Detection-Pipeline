"""
Test script for OCR engines
Quick verification that all engines work correctly
"""

import cv2
import numpy as np
from ocr_engines import TrOCREngine, PaddleOCREngine, TesseractEngine

def create_test_image():
    """Create a simple test image with text"""
    # Create white background
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    
    # Add text
    text = "WAGON 12345"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (50, 100), font, 2, (0, 0, 0), 3)
    
    return img

def test_engine(engine_class, engine_name, **kwargs):
    """Test a single OCR engine"""
    print(f"\n{'='*60}")
    print(f"Testing {engine_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize engine
        print(f"Initializing {engine_name}...")
        engine = engine_class(**kwargs)
        
        if not engine.initialize():
            print(f"❌ Failed to initialize {engine_name}")
            return False
        
        print(f"✅ {engine_name} initialized successfully")
        
        # Create test image
        test_image = create_test_image()
        
        # Run OCR
        print(f"Running OCR...")
        result = engine.extract_text(test_image)
        
        # Display results
        if result['success']:
            print(f"✅ OCR completed successfully")
            print(f"   Text: '{result['text']}'")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            print(f"   Bounding boxes: {len(result['boxes'])}")
            return True
        else:
            print(f"❌ OCR failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Run tests for all OCR engines"""
    print("="*60)
    print("OCR ENGINE TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test TrOCR
    results['TrOCR'] = test_engine(
        TrOCREngine,
        "TrOCR (Transformer)"
    )
    
    # Test PaddleOCR 2.7.x
    results['PaddleOCR'] = test_engine(
        PaddleOCREngine,
        "PaddleOCR 2.7.x",
        use_gpu=False  # Use CPU for testing
    )
    
    # Test Tesseract
    results['Tesseract'] = test_engine(
        TesseractEngine,
        "Tesseract OCR",
        digits_only=False
    )
    
    # Test Tesseract (digits only)
    results['Tesseract (Digits)'] = test_engine(
        TesseractEngine,
        "Tesseract OCR (Digits Only)",
        digits_only=True
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for engine, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {engine}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} engines working")
    
    if passed == total:
        print("\n🎉 All OCR engines are working correctly!")
    elif passed > 0:
        print(f"\n⚠️  {total - passed} engine(s) failed. Check installation.")
    else:
        print("\n❌ No OCR engines are working. Please install dependencies.")
    
    print("\nInstallation commands:")
    print("  TrOCR:      pip install transformers torch")
    print("  PaddleOCR:  pip install paddleocr==2.7.0.3")
    print("  Tesseract:  pip install pytesseract + binary installation")

if __name__ == "__main__":
    main()
