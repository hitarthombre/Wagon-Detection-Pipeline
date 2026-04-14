"""
Test script to verify all enhancements are working correctly.
Run this to quickly test the new features.
"""

import os
import sys

def test_file_exists(filepath):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {filepath} exists")
        return True
    else:
        print(f"❌ {filepath} NOT FOUND")
        return False

def test_imports():
    """Test if all required imports are available"""
    print("\n=== Testing Imports ===")
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO available")
    except ImportError:
        print("❌ Ultralytics not installed")
        return False
    
    try:
        import easyocr
        print("✅ EasyOCR available")
    except ImportError:
        print("❌ EasyOCR not installed")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not installed")
        return False
    
    return True

def test_files():
    """Test if all required files exist"""
    print("\n=== Testing Files ===")
    
    files = [
        "step4_object_detection.py",
        "step5_ocr_extraction.py",
        "integrated_pipeline.py",
        "README_ENHANCEMENTS.md",
        "QUICKSTART_ENHANCED.md",
        "CHANGES_SUMMARY.md",
        "yolov8n.pt"
    ]
    
    all_exist = True
    for file in files:
        if not test_file_exists(file):
            all_exist = False
    
    return all_exist

def test_video_files():
    """Test if video files exist"""
    print("\n=== Testing Video Files ===")
    
    if not os.path.exists("video"):
        print("❌ video/ folder not found")
        return False
    
    video_files = [f for f in os.listdir("video") if f.endswith(".mp4")]
    
    if len(video_files) == 0:
        print("❌ No video files found in video/ folder")
        return False
    
    print(f"✅ Found {len(video_files)} video file(s):")
    for vf in video_files:
        print(f"   - {vf}")
    
    return True

def test_output_folder():
    """Test if output folder can be created"""
    print("\n=== Testing Output Folder ===")
    
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"✅ Created {output_dir}/ folder")
            return True
        except Exception as e:
            print(f"❌ Failed to create {output_dir}/ folder: {e}")
            return False
    else:
        print(f"✅ {output_dir}/ folder already exists")
        return True

def test_code_syntax():
    """Test if Python files have valid syntax"""
    print("\n=== Testing Code Syntax ===")
    
    files = [
        "step4_object_detection.py",
        "step5_ocr_extraction.py",
        "integrated_pipeline.py"
    ]
    
    all_valid = True
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, file, 'exec')
            print(f"✅ {file} - syntax OK")
        except SyntaxError as e:
            print(f"❌ {file} - syntax error: {e}")
            all_valid = False
        except Exception as e:
            print(f"❌ {file} - error: {e}")
            all_valid = False
    
    return all_valid

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    print("\n1. Run integrated pipeline (recommended):")
    print("   python integrated_pipeline.py video/video_test_1.mp4")
    print("\n2. Run object detection only:")
    print("   python step4_object_detection.py video/video_test_1.mp4")
    print("\n3. Run OCR only:")
    print("   python step5_ocr_extraction.py video/video_test_1.mp4")
    print("\nKeyboard Controls:")
    print("   'p' - Pause video")
    print("   'r' - Resume video")
    print("   'q' - Quit")
    print("\nOutput:")
    print("   Results will be saved to output/ folder")
    print("="*80)

def main():
    """Run all tests"""
    print("="*80)
    print("ENHANCEMENT TEST SUITE")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Files", test_files()))
    results.append(("Video Files", test_video_files()))
    results.append(("Output Folder", test_output_folder()))
    results.append(("Code Syntax", test_code_syntax()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        print_usage_instructions()
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
