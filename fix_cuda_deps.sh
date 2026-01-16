#!/bin/bash
# ============================================================
# Fix CUDA Dependencies Conflict
# Run this after setup_gpu.sh if you see CUDA library conflicts
# ============================================================

echo "============================================================"
echo "Fixing CUDA dependency conflicts..."
echo "============================================================"

# Activate virtual environment
source venv/bin/activate

# Reinstall PyTorch's specific CUDA libraries
echo "Reinstalling PyTorch CUDA libraries..."
pip install --force-reinstall \
    nvidia-cublas-cu12==12.4.5.8 \
    nvidia-cuda-cupti-cu12==12.4.127 \
    nvidia-cuda-nvrtc-cu12==12.4.127 \
    nvidia-cuda-runtime-cu12==12.4.127 \
    nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-cufft-cu12==11.2.1.3 \
    nvidia-curand-cu12==10.3.5.147 \
    nvidia-cusolver-cu12==11.6.1.9 \
    nvidia-cusparse-cu12==12.3.1.170 \
    nvidia-nvjitlink-cu12==12.4.127

echo ""
echo "Verifying installations..."

python3 << 'EOF'
import sys
print("=" * 60)
print("Verification Results")
print("=" * 60)

# Test TensorFlow
print("\n1. TensorFlow GPU:")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   [OK] {len(gpus)} GPU(s) detected")
        for gpu in gpus:
            print(f"        - {gpu}")
    else:
        print("   [WARNING] No GPU detected")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test ONNX Runtime
print("\n2. ONNX Runtime GPU:")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        print(f"   [OK] CUDA provider available")
    else:
        print(f"   [WARNING] CUDA provider not available")
    print(f"   Available: {providers}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test PyTorch
print("\n3. PyTorch CUDA:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   [OK] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("   [WARNING] CUDA not available")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test InsightFace
print("\n4. InsightFace:")
try:
    from insightface.app import FaceAnalysis
    print("   [OK] InsightFace imported successfully")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test DeepFace
print("\n5. DeepFace:")
try:
    from deepface import DeepFace
    print("   [OK] DeepFace imported successfully")
except Exception as e:
    print(f"   [ERROR] {e}")

print("\n" + "=" * 60)
EOF

echo ""
echo "Fix complete. You can now run: ./run_gpu.sh"
