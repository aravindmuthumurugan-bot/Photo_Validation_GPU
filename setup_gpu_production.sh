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

# Get CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1 | cut -d'.' -f1,2)
print_status "CUDA version: $CUDA_VERSION"

# Set CUDA environment variables
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

# Step 3: Install TensorFlow FIRST (before anything else)
echo ""
echo "Step 3: Installing TensorFlow with CUDA support..."
echo "------------------------------------------------------------"

pip install tensorflow==2.16.2
pip install tf-keras
print_status "TensorFlow installed"

# Step 4: Install ONNX Runtime GPU
echo ""
echo "Step 4: Installing ONNX Runtime GPU..."
echo "------------------------------------------------------------"

pip install onnxruntime-gpu==1.19.2

# Verify ONNX Runtime CUDA immediately
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'ONNX Runtime providers: {providers}')
if 'CUDAExecutionProvider' not in providers:
    print('ERROR: CUDA provider not available!')
    exit(1)
print('ONNX Runtime CUDA: OK')
"
print_status "ONNX Runtime GPU installed and verified"

# Step 5: Install InsightFace (does NOT depend on onnxruntime, uses onnxruntime-gpu)
echo ""
echo "Step 5: Installing InsightFace..."
echo "------------------------------------------------------------"

pip install insightface==0.7.3
print_status "InsightFace installed"

# Step 6: Install DeepFace and dependencies (SKIP mtcnn/retina-face auto-install)
echo ""
echo "Step 6: Installing DeepFace..."
echo "------------------------------------------------------------"

# Install DeepFace without its face detector dependencies first
pip install --no-deps deepface==0.0.93

# Install DeepFace dependencies manually (excluding those that pull onnxruntime)
pip install flask flask-cors fire gunicorn gdown

print_status "DeepFace installed"

# Step 7: Install retina-face and mtcnn carefully
echo ""
echo "Step 7: Installing face detectors..."
echo "------------------------------------------------------------"

# Install retina-face without dependencies, then add what it needs
pip install --no-deps retina-face==0.0.17
pip install --no-deps mtcnn==1.0.0
pip install lz4 joblib

print_status "Face detectors installed"

# Step 8: Install NudeNet WITHOUT its onnxruntime dependency
echo ""
echo "Step 8: Installing NudeNet (without CPU onnxruntime)..."
echo "------------------------------------------------------------"

# Install nudenet without dependencies
pip install --no-deps nudenet==3.4.2

print_status "NudeNet installed"

# Step 9: Verify ONNX Runtime GPU is still intact
echo ""
echo "Step 9: Verifying ONNX Runtime GPU is still available..."
echo "------------------------------------------------------------"

python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'ONNX Runtime providers: {providers}')
if 'CUDAExecutionProvider' not in providers:
    print('ERROR: CUDA provider was overwritten!')
    exit(1)
print('ONNX Runtime CUDA: Still OK')
"
print_status "ONNX Runtime GPU verified"

# Step 10: Install remaining dependencies
echo ""
echo "Step 10: Installing remaining dependencies..."
echo "------------------------------------------------------------"

pip install \
    opencv-python-headless \
    pillow \
    scikit-image \
    scikit-learn \
    albumentations

pip install \
    fastapi==0.115.6 \
    "uvicorn[standard]==0.34.0" \
    python-multipart==0.0.20 \
    pydantic

pip install \
    pandas \
    scipy \
    requests \
    tqdm \
    PyYAML \
    matplotlib \
    easydict \
    cython \
    prettytable

print_status "All dependencies installed"

# Step 11: Download InsightFace models
echo ""
echo "Step 11: Downloading InsightFace models..."
echo "------------------------------------------------------------"

python3 << 'EOF'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Downloading InsightFace buffalo_l model...")
from insightface.app import FaceAnalysis

# Download with CPU first to avoid issues during download
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("Model downloaded successfully")
EOF

print_status "Models downloaded"

# Step 12: Final verification
echo ""
echo "Step 12: Final GPU verification..."
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

# 6. NudeNet
print("\n6. NudeNet:")
try:
    from nudenet import NudeDetector
    print("   [OK] NudeNet imported")
except Exception as e:
    print(f"   [FAIL] {e}")
    errors.append(f"NudeNet: {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print("VERIFICATION FAILED - NOT READY FOR PRODUCTION")
    print("=" * 60)
    for err in errors:
        print(f"  - {err}")
    exit(1)
else:
    print("ALL CHECKS PASSED - READY FOR PRODUCTION")
    print("=" * 60)
    print("  - ONNX Runtime: GPU (CUDA)")
    print("  - TensorFlow: GPU (CUDA)")
    print("  - InsightFace: GPU")
    print("  - DeepFace: GPU (via TensorFlow)")
    print("  - RetinaFace: OK")
    print("  - NudeNet: OK")
EOF

if [ $? -ne 0 ]; then
    print_error "GPU verification failed!"
    exit 1
fi

# Step 13: Create production startup script
echo ""
echo "Step 13: Creating production startup script..."
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
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

# Activate venv
source venv/bin/activate

echo "============================================================"
echo "Starting Photo Validation API - PRODUCTION MODE"
echo "============================================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $CUDA_HOME"
echo "============================================================"

# Run with uvicorn
exec uvicorn api_hybrid:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 30
RUNSCRIPT

chmod +x run_production.sh
print_status "Production startup script created"

# Create temp directory
mkdir -p /tmp/photo_uploads
chmod 755 /tmp/photo_uploads

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
echo "  - NudeNet: ONNX Runtime CUDA"
echo ""
echo "To start the production server:"
echo "  ./run_production.sh"
echo ""
echo "API endpoints:"
echo "  - Health: http://0.0.0.0:8000/health"
echo "  - Docs:   http://0.0.0.0:8000/docs"
echo "============================================================"
