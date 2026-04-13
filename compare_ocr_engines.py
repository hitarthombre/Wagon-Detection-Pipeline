"""
Compare EasyOCR vs PaddleOCR performance on a sample video.
This script helps you choose the best OCR engine for your use case.
"""

import cv2
import time
import sys
import os

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR not installed. Install with: pip install easyocr")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("⚠️  PaddleOCR not installed. Install with: pip install paddleocr")

def test_ocr_engine(video_path, engine_name, ocr_engine, num_frames=10):
    """Test OCR engine on sample frames"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    print(f"\nTesting {engine_name}...")
    print(f"Processing {num_frames} frames...")
    
    total_time = 0
    total_text = 0
    frame_count = 0
    
    for i in range(num_frames):
        # Seek to frame
        frame_idx = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Time OCR processing
        start_time = time.time()
        
        if engine_name == "PaddleOCR":
            result = ocr_engine.predict(frame)
            text_count = 0
            if result and result.get('rec_text'):
                text_count = len(result['rec_text'])
        else:  # EasyOCR
            result = ocr_engine.readtext(frame)
            text_count = len(result)
        
        elapsed = time.time() - start_time
        
        total_time += elapsed
        total_text += text_count
        frame_count += 1
        
        print(f"  Frame {frame_idx}: {text_count} texts in {elapsed:.2f}s")
    
    cap.release()
    
    if frame_count == 0:
        return None
    
    avg_time = total_time / frame_count
    avg_text = total_text / frame_count
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    return {
        'engine': engine_name,
        'frames': frame_count,
        'total_time': total_time,
        'avg_time': avg_time,
        'total_text': total_text,
        'avg_text': avg_text,
        'fps': fps
    }

def main():
    if len(sys.argv) < 2:
        default_video = "video/video_test_1.mp4"
        if os.path.exists(default_video):
            print(f"Using default video: {default_video}")
            video_path = default_video
        else:
            print("Usage: python compare_ocr_engines.py <video_path> [num_frames]")
            print("Example: python compare_ocr_engines.py video/video_test_1.mp4 10")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print("="*80)
    print("OCR ENGINE COMPARISON")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Test frames: {num_frames}")
    
    results = []
    
    # Test EasyOCR
    if EASYOCR_AVAILABLE:
        print("\n" + "-"*80)
        print("Initializing EasyOCR...")
        easy_ocr = easyocr.Reader(['en'], gpu=True)
        result = test_ocr_engine(video_path, "EasyOCR", easy_ocr, num_frames)
        if result:
            results.append(result)
    
    # Test PaddleOCR
    if PADDLEOCR_AVAILABLE:
        print("\n" + "-"*80)
        print("Initializing PaddleOCR...")
        paddle_ocr = None
        try:
            # PaddleOCR 3.0+ simplified API
            paddle_ocr = PaddleOCR(lang='en')
            print("Using PaddleOCR 3.0+ API")
        except Exception as e:
            print(f"PaddleOCR initialization failed: {e}")
            print("PaddleOCR not compatible with this version")
        
        if paddle_ocr:
            result = test_ocr_engine(video_path, "PaddleOCR", paddle_ocr, num_frames)
            if result:
                results.append(result)
    
    # Display comparison
    if len(results) == 0:
        print("\n❌ No OCR engines available for testing.")
        print("Install at least one:")
        print("  pip install easyocr")
        print("  pip install paddleocr")
        return
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n{result['engine']}:")
        print(f"  Frames processed: {result['frames']}")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Average time per frame: {result['avg_time']:.2f}s")
        print(f"  Average FPS: {result['fps']:.2f}")
        print(f"  Total text detected: {result['total_text']}")
        print(f"  Average text per frame: {result['avg_text']:.2f}")
    
    # Winner
    if len(results) == 2:
        print("\n" + "="*80)
        print("WINNER")
        print("="*80)
        
        faster = results[0] if results[0]['fps'] > results[1]['fps'] else results[1]
        speedup = max(results[0]['fps'], results[1]['fps']) / min(results[0]['fps'], results[1]['fps'])
        
        print(f"\n🏆 {faster['engine']} is {speedup:.1f}x faster!")
        print(f"   {faster['engine']}: {faster['fps']:.2f} FPS")
        
        if results[0]['total_text'] != results[1]['total_text']:
            more_text = results[0] if results[0]['total_text'] > results[1]['total_text'] else results[1]
            print(f"\n📝 {more_text['engine']} detected more text:")
            print(f"   {results[0]['engine']}: {results[0]['total_text']} texts")
            print(f"   {results[1]['engine']}: {results[1]['total_text']} texts")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if len(results) == 2:
        if faster['engine'] == 'PaddleOCR':
            print("\n✅ Use PaddleOCR for:")
            print("   - Faster processing (2-3x speed)")
            print("   - Better accuracy on complex text")
            print("   - Production environments")
            print("\n   Command:")
            print(f"   python step5_ocr_extraction.py {video_path} 100 True True paddleocr")
        else:
            print("\n✅ Use EasyOCR for:")
            print("   - Maximum compatibility")
            print("   - Specific language requirements")
            print("   - When PaddleOCR is not available")
            print("\n   Command:")
            print(f"   python step5_ocr_extraction.py {video_path} 100 True True easyocr")
    else:
        engine = results[0]['engine'].lower()
        print(f"\n✅ Using {results[0]['engine']} (only engine available)")
        print(f"\n   Command:")
        print(f"   python step5_ocr_extraction.py {video_path} 100 True True {engine}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
