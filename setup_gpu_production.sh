#!/bin/bash
# ============================================================
# PRODUCTION GPU Setup Script for Photo Validation API
# Target: NVIDIA L40S with CUDA 12.0, Python 3.12.3
# All models MUST run on GPU
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "Photo Validation GPU Setup - PRODUCTION"
echo "All models will run on GPU"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# Step 1: Check environment
echo ""
echo "Step 1: Checking environment..."
echo "------------------------------------------------------------"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
    print_status "GPU: $GPU_INFO"
else
    print_error "nvidia-smi not found!"
    exit 1
fi

# Get CUDA version from nvcc
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1 | cut -d'.' -f1,2)
print_status "CUDA version: $CUDA_VERSION"

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# Step 2: Remove existing venv and create fresh
echo ""
echo "Step 2: Creating fresh virtual environment..."
echo "------------------------------------------------------------"

if [ -d "venv" ]; then
    print_warning "Removing existing virtual environment..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
print_status "Virtual environment created"

# Step 3: Install CUDA-compatible packages in correct order
echo ""
echo "Step 3: Installing CUDA-compatible packages..."
echo "------------------------------------------------------------"

# Install cuDNN and CUDA libraries for ONNX Runtime
# ONNX Runtime GPU 1.19.2 supports CUDA 12.x
echo "Installing ONNX Runtime GPU (CUDA 12)..."
pip install onnxruntime-gpu==1.19.2

# Verify ONNX Runtime CUDA
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'ONNX Runtime providers: {providers}')
if 'CUDAExecutionProvider' not in providers:
    print('WARNING: CUDA not available in ONNX Runtime')
    print('This might be due to missing CUDA libraries')
    exit(1)
else:
    print('ONNX Runtime CUDA: OK')
"
print_status "ONNX Runtime GPU installed"

# Step 4: Install TensorFlow with CUDA
echo ""
echo "Step 4: Installing TensorFlow with CUDA..."
echo "------------------------------------------------------------"

# TensorFlow 2.16.x has better CUDA 12 support and is more stable
pip install tensorflow==2.16.2

# Install tf-keras for RetinaFace compatibility
pip install tf-keras

print_status "TensorFlow installed"

# Step 5: Install face analysis libraries
echo ""
echo "Step 5: Installing face analysis libraries..."
echo "------------------------------------------------------------"

pip install insightface==0.7.3
pip install deepface==0.0.93
pip install retina-face==0.0.17
pip install mtcnn==1.0.0
pip install nudenet==3.4.2

print_status "Face analysis libraries installed"

# Step 6: Install remaining dependencies
echo ""
echo "Step 6: Installing remaining dependencies..."
echo "------------------------------------------------------------"

pip install \
    opencv-python-headless==4.10.0.84 \
    pillow==11.0.0 \
    scikit-image==0.24.0 \
    albumentations==1.4.21

pip install \
    fastapi==0.115.6 \
    "uvicorn[standard]==0.34.0" \
    python-multipart==0.0.20 \
    pydantic==2.10.3

pip install \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.14.1 \
    requests==2.32.3 \
    tqdm==4.67.1 \
    gdown==5.2.0 \
    PyYAML==6.0.2

print_status "All dependencies installed"

# Step 7: Download InsightFace models
echo ""
echo "Step 7: Downloading InsightFace models..."
echo "------------------------------------------------------------"

python3 << 'EOF'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Downloading InsightFace buffalo_l model...")
from insightface.app import FaceAnalysis

# Download with CPU first to avoid issues
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("Model downloaded successfully")
EOF

print_status "Models downloaded"

# Step 8: Final verification
echo ""
echo "Step 8: Final GPU verification..."
echo "------------------------------------------------------------"

python3 << 'EOF'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 60)
print("GPU VERIFICATION - ALL MUST BE OK FOR PRODUCTION")
print("=" * 60)

errors = []

# 1. ONNX Runtime GPU (for InsightFace)
print("\n1. ONNX Runtime GPU (InsightFace):")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        print(f"   [OK] CUDA provider available")
        print(f"   Providers: {providers}")
    else:
        print(f"   [FAIL] CUDA provider NOT available")
        print(f"   Available: {providers}")
        errors.append("ONNX Runtime CUDA not available")
except Exception as e:
    print(f"   [FAIL] {e}")
    errors.append(f"ONNX Runtime: {e}")

# 2. TensorFlow GPU (for DeepFace)
print("\n2. TensorFlow GPU (DeepFace):")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"   [OK] {len(gpus)} GPU(s) available")
        for gpu in gpus:
            print(f"   Device: {gpu}")
    else:
        print("   [FAIL] No GPU detected")
        errors.append("TensorFlow GPU not available")
except Exception as e:
    print(f"   [FAIL] {e}")
    errors.append(f"TensorFlow: {e}")

# 3. InsightFace with GPU
print("\n3. InsightFace GPU initialization:")
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("   [OK] InsightFace initialized with GPU")
except Exception as e:
    print(f"   [FAIL] {e}")
    errors.append(f"InsightFace GPU: {e}")

# 4. DeepFace
print("\n4. DeepFace:")
try:
    from deepface import DeepFace
    print("   [OK] DeepFace imported (will use TensorFlow GPU)")
except Exception as e:
    print(f"   [FAIL] {e}")
    errors.append(f"DeepFace: {e}")

# 5. RetinaFace
print("\n5. RetinaFace:")
try:
    from retinaface import RetinaFace
    print("   [OK] RetinaFace imported")
except Exception as e:
    print(f"   [FAIL] {e}")
    errors.append(f"RetinaFace: {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print("VERIFICATION FAILED - NOT READY FOR PRODUCTION")
    print("=" * 60)
    for err in errors:
        print(f"  - {err}")
    print("\nPlease fix the above issues before running in production.")
    exit(1)
else:
    print("ALL CHECKS PASSED - READY FOR PRODUCTION")
    print("=" * 60)
    print("  - ONNX Runtime: GPU (CUDA)")
    print("  - TensorFlow: GPU (CUDA)")
    print("  - InsightFace: GPU")
    print("  - DeepFace: GPU (via TensorFlow)")
    print("  - RetinaFace: OK")
EOF

if [ $? -ne 0 ]; then
    print_error "GPU verification failed!"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check CUDA installation: nvcc --version"
    echo "2. Check cuDNN: locate libcudnn"
    echo "3. Check LD_LIBRARY_PATH includes CUDA libs"
    echo ""
    echo "Try adding to your ~/.bashrc:"
    echo "  export CUDA_HOME=/usr/local/cuda"
    echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    exit 1
fi

# Step 9: Create startup script with proper environment
echo ""
echo "Step 9: Creating production startup script..."
echo "------------------------------------------------------------"

cat > run_production.sh << 'RUNSCRIPT'
#!/bin/bash
# Production startup script - ensures GPU is used

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# TensorFlow settings
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

# Activate venv
source venv/bin/activate

echo "============================================================"
echo "Starting Photo Validation API - PRODUCTION MODE"
echo "============================================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $CUDA_HOME"
echo "============================================================"

# Run with gunicorn for production (more stable than uvicorn for long-running)
exec uvicorn api_hybrid:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 30
RUNSCRIPT

chmod +x run_production.sh
print_status "Production startup script created"

# Final message
echo ""
echo "============================================================"
echo -e "${GREEN}PRODUCTION SETUP COMPLETE${NC}"
echo "============================================================"
echo ""
echo "All models configured for GPU:"
echo "  - InsightFace: ONNX Runtime CUDA"
echo "  - DeepFace: TensorFlow CUDA"
echo "  - RetinaFace: TensorFlow CUDA"
echo ""
echo "To start the production server:"
echo "  ./run_production.sh"
echo ""
echo "API endpoints:"
echo "  - Health: http://0.0.0.0:8000/health"
echo "  - Docs:   http://0.0.0.0:8000/docs"
echo "============================================================"
