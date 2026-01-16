#!/usr/bin/env python3
"""
GPU Test Script - Verify all components are working
Run this before starting the API server
"""

import sys
import os

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def test_tensorflow():
    """Test TensorFlow GPU"""
    print("\n" + "="*60)
    print("1. Testing TensorFlow GPU")
    print("="*60)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[OK] TensorFlow: {len(gpus)} GPU(s) detected")
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"     - {details.get('device_name', gpu)}")
            return True
        else:
            print("[WARNING] TensorFlow: No GPU detected, will use CPU")
            return False
    except Exception as e:
        print(f"[ERROR] TensorFlow: {e}")
        return False

def test_onnxruntime():
    """Test ONNX Runtime GPU"""
    print("\n" + "="*60)
    print("2. Testing ONNX Runtime GPU")
    print("="*60)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        if 'CUDAExecutionProvider' in providers:
            print("[OK] ONNX Runtime: CUDA provider available")
            return True
        else:
            print("[WARNING] ONNX Runtime: CUDA provider not available, will use CPU")
            return False
    except Exception as e:
        print(f"[ERROR] ONNX Runtime: {e}")
        return False

def test_pytorch():
    """Test PyTorch CUDA"""
    print("\n" + "="*60)
    print("3. Testing PyTorch CUDA")
    print("="*60)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] PyTorch CUDA available")
            print(f"     Device: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA version: {torch.version.cuda}")
            print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("[WARNING] PyTorch: CUDA not available")
            return False
    except Exception as e:
        print(f"[ERROR] PyTorch: {e}")
        return False

def test_insightface():
    """Test InsightFace initialization"""
    print("\n" + "="*60)
    print("4. Testing InsightFace")
    print("="*60)
    try:
        from insightface.app import FaceAnalysis
        print("[OK] InsightFace imported successfully")

        # Try to initialize (will download models if needed)
        print("     Initializing FaceAnalysis (may download models)...")
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[OK] InsightFace initialized with GPU")
        return True
    except Exception as e:
        print(f"[WARNING] InsightFace GPU init failed: {e}")
        try:
            app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            print("[OK] InsightFace initialized with CPU fallback")
            return True
        except Exception as e2:
            print(f"[ERROR] InsightFace: {e2}")
            return False

def test_deepface():
    """Test DeepFace"""
    print("\n" + "="*60)
    print("5. Testing DeepFace")
    print("="*60)
    try:
        from deepface import DeepFace
        print("[OK] DeepFace imported successfully")
        return True
    except Exception as e:
        print(f"[ERROR] DeepFace: {e}")
        return False

def test_other_dependencies():
    """Test other critical dependencies"""
    print("\n" + "="*60)
    print("6. Testing Other Dependencies")
    print("="*60)

    deps = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('retinaface', 'RetinaFace'),
        ('nudenet', 'NudeNet'),
    ]

    all_ok = True
    for module, name in deps:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            all_ok = False

    return all_ok

def main():
    print("="*60)
    print("Photo Validation GPU Test Suite")
    print("="*60)

    results = {
        'TensorFlow GPU': test_tensorflow(),
        'ONNX Runtime GPU': test_onnxruntime(),
        'PyTorch CUDA': test_pytorch(),
        'InsightFace': test_insightface(),
        'DeepFace': test_deepface(),
        'Other Dependencies': test_other_dependencies(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    gpu_ready = False
    for name, status in results.items():
        status_str = "[OK]" if status else "[FAIL]"
        print(f"{status_str} {name}")
        if name in ['TensorFlow GPU', 'ONNX Runtime GPU'] and status:
            gpu_ready = True

    print("\n" + "="*60)
    if gpu_ready:
        print("GPU MODE: Ready to run with GPU acceleration!")
    else:
        print("CPU MODE: Will run on CPU (slower but functional)")
    print("="*60)

    # Check if critical components work
    critical = ['InsightFace', 'DeepFace', 'Other Dependencies']
    if all(results.get(c, False) for c in critical):
        print("\nAll critical components are working.")
        print("You can start the server with: ./run_gpu.sh")
        return 0
    else:
        print("\nSome critical components failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
