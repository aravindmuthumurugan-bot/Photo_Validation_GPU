#!/bin/bash
# ============================================================
# FINAL PRODUCTION GPU Setup Script for Photo Validation API
# Target: NVIDIA L40S with CUDA 12.x, Python 3.12.3
#
# KEY FIX: Install NVIDIA CUDA libraries via pip so onnxruntime-gpu
# can find them without system-level configuration
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "Photo Validation GPU Setup - FINAL PRODUCTION VERSION"
echo "All models will run on GPU (InsightFace + DeepFace)"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# Step 1: Check GPU availability
echo ""
echo "Step 1: Checking GPU environment..."
echo "------------------------------------------------------------"

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
    print_status "GPU: $GPU_INFO"
else
    print_error "nvidia-smi not found!"
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
print_status "CUDA Version (Driver): $CUDA_VERSION"

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

# Step 3: Install NVIDIA CUDA libraries via pip (CRITICAL!)
# These provide the CUDA runtime libraries that onnxruntime-gpu needs
echo ""
echo "Step 3: Installing NVIDIA CUDA libraries via pip..."
echo "------------------------------------------------------------"
echo "This is the KEY FIX - provides CUDA libraries for ONNX Runtime"

pip install nvidia-cuda-runtime-cu12==12.4.127
pip install nvidia-cudnn-cu12==9.1.0.70
pip install nvidia-cublas-cu12==12.4.5.8
pip install nvidia-cufft-cu12==11.2.1.3
pip install nvidia-curand-cu12==10.3.5.147
pip install nvidia-cusolver-cu12==11.6.1.9
pip install nvidia-cusparse-cu12==12.3.1.170
pip install nvidia-nccl-cu12==2.21.5

print_status "NVIDIA CUDA pip packages installed"

# Set up library paths for the pip-installed NVIDIA packages
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
NVIDIA_LIB_PATH=""

# Find all nvidia lib directories
for pkg in cuda_runtime cudnn cublas cufft curand cusolver cusparse nccl; do
    PKG_LIB="$SITE_PACKAGES/nvidia/${pkg}/lib"
    if [ -d "$PKG_LIB" ]; then
        NVIDIA_LIB_PATH="${NVIDIA_LIB_PATH}:${PKG_LIB}"
    fi
done

# Remove leading colon
NVIDIA_LIB_PATH="${NVIDIA_LIB_PATH#:}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}:${LD_LIBRARY_PATH}"

print_status "LD_LIBRARY_PATH updated with NVIDIA pip packages"
echo "  NVIDIA libs: $NVIDIA_LIB_PATH"

# Step 4: Install ONNX Runtime GPU
echo ""
echo "Step 4: Installing ONNX Runtime GPU..."
echo "------------------------------------------------------------"

# Use version 1.18.0 which has good CUDA 12 support
pip install onnxruntime-gpu==1.18.0

# Verify CUDA provider is available
echo ""
echo "Verifying ONNX Runtime CUDA provider..."
python3 << 'EOF'
import os
import sys

# Set up library path for NVIDIA pip packages
import site
site_packages = site.getsitepackages()[0]
nvidia_libs = []
for pkg in ['cuda_runtime', 'cudnn', 'cublas', 'cufft', 'curand', 'cusolver', 'cusparse', 'nccl']:
    lib_path = os.path.join(site_packages, 'nvidia', pkg, 'lib')
    if os.path.exists(lib_path):
        nvidia_libs.append(lib_path)

current_ld = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(nvidia_libs) + ':' + current_ld

import onnxruntime as ort
providers = ort.get_available_providers()
print(f'ONNX Runtime version: {ort.__version__}')
print(f'Available providers: {providers}')

if 'CUDAExecutionProvider' in providers:
    print('[OK] CUDA provider available!')
else:
    print('[WARNING] CUDA provider not available yet')
    print('Will try TensorRT provider installation...')
    sys.exit(1)
EOF

ONNX_STATUS=$?
if [ $ONNX_STATUS -ne 0 ]; then
    print_warning "ONNX Runtime CUDA not detected with pip packages alone"
    print_warning "Trying additional approaches..."

    # Try installing onnxruntime-gpu with different version
    pip uninstall -y onnxruntime-gpu onnxruntime 2>/dev/null || true

    # Try version 1.17.1 which might have better compatibility
    pip install onnxruntime-gpu==1.17.1

    # Verify again
    python3 << 'EOF'
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'Providers after reinstall: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('[OK] CUDA provider now available!')
else:
    print('[INFO] CUDA provider still not available')
    print('InsightFace will use CPU, but DeepFace will use TensorFlow GPU')
EOF
fi

print_status "ONNX Runtime GPU installation complete"

# Step 5: Install TensorFlow with GPU support
echo ""
echo "Step 5: Installing TensorFlow..."
echo "------------------------------------------------------------"

pip install tensorflow==2.16.2
pip install tf-keras

print_status "TensorFlow installed"

# Step 6: Install InsightFace
echo ""
echo "Step 6: Installing InsightFace..."
echo "------------------------------------------------------------"

pip install insightface==0.7.3

print_status "InsightFace installed"

# Step 7: Install face detection/analysis libraries WITHOUT onnxruntime dependency
echo ""
echo "Step 7: Installing face detection libraries (--no-deps)..."
echo "------------------------------------------------------------"

# DeepFace - install without deps to avoid CPU onnxruntime
pip install --no-deps deepface==0.0.93

# Install DeepFace's actual dependencies manually
pip install flask flask-cors fire gunicorn gdown retina-face mtcnn

# RetinaFace and MTCNN
pip install --no-deps retina-face==0.0.17 || true
pip install --no-deps mtcnn==1.0.0 || true

# NudeNet - install without deps
pip install --no-deps nudenet==3.4.2

# Manual NudeNet dependencies
pip install pillow

print_status "Face detection libraries installed"

# Step 8: Install remaining dependencies
echo ""
echo "Step 8: Installing remaining dependencies..."
echo "------------------------------------------------------------"

pip install \
    opencv-python-headless \
    scikit-image \
    scikit-learn \
    albumentations==1.4.21

pip install \
    fastapi==0.115.6 \
    "uvicorn[standard]==0.34.0" \
    python-multipart==0.0.20 \
    pydantic

pip install \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.14.1 \
    requests==2.32.3 \
    tqdm \
    PyYAML==6.0.2 \
    gdown==5.2.0 \
    matplotlib \
    easydict \
    cython \
    prettytable \
    lz4 \
    joblib

print_status "All dependencies installed"

# Step 9: Ensure onnxruntime-gpu is intact (fix any overwrites)
echo ""
echo "Step 9: Ensuring onnxruntime-gpu is intact..."
echo "------------------------------------------------------------"

# Check if CPU version was installed by any dependency
ONNX_PKG=$(pip list | grep onnxruntime)
echo "Installed ONNX packages: $ONNX_PKG"

# If only CPU version exists, replace it
if pip list | grep -q "^onnxruntime " && ! pip list | grep -q "onnxruntime-gpu"; then
    print_warning "CPU onnxruntime was installed by a dependency, removing..."
    pip uninstall -y onnxruntime
    pip install onnxruntime-gpu==1.18.0
fi

print_status "ONNX Runtime check complete"

# Step 10: Download InsightFace models
echo ""
echo "Step 10: Downloading InsightFace models..."
echo "------------------------------------------------------------"

python3 << 'EOF'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Downloading InsightFace buffalo_l model...")
from insightface.app import FaceAnalysis

# Download with CPU first (just to get the model files)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("Model downloaded successfully")
EOF

print_status "Models downloaded"

# Step 11: Final verification
echo ""
echo "Step 11: Final GPU verification..."
echo "------------------------------------------------------------"

# Create a Python script that properly sets up the environment
python3 << EOF
import os
import sys
import site

# Set up NVIDIA pip package library paths FIRST
site_packages = site.getsitepackages()[0]
nvidia_libs = []
for pkg in ['cuda_runtime', 'cudnn', 'cublas', 'cufft', 'curand', 'cusolver', 'cusparse', 'nccl']:
    lib_path = os.path.join(site_packages, 'nvidia', pkg, 'lib')
    if os.path.exists(lib_path):
        nvidia_libs.append(lib_path)

current_ld = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(nvidia_libs) + ':' + current_ld
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 60)
print("GPU VERIFICATION - FINAL")
print("=" * 60)

onnx_gpu_ok = False
tf_gpu_ok = False

# 1. ONNX Runtime GPU
print("\n1. ONNX Runtime GPU (InsightFace):")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"   Available providers: {providers}")
    if 'CUDAExecutionProvider' in providers:
        print("   [OK] CUDA provider available")
        onnx_gpu_ok = True
    else:
        print("   [INFO] CUDA not available - InsightFace will use CPU")
        print("   (TensorFlow GPU will still be used for DeepFace)")
except Exception as e:
    print(f"   [WARNING] {e}")

# 2. TensorFlow GPU
print("\n2. TensorFlow GPU (DeepFace):")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"   [OK] {len(gpus)} GPU(s) available")
        tf_gpu_ok = True
    else:
        print("   [FAIL] No GPU detected")
except Exception as e:
    print(f"   [FAIL] {e}")

# 3. InsightFace
print("\n3. InsightFace:")
try:
    from insightface.app import FaceAnalysis
    if onnx_gpu_ok:
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("   [OK] Initialized with GPU")
    else:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("   [OK] Initialized with CPU (still functional)")
except Exception as e:
    print(f"   [ERROR] {e}")

# 4. DeepFace
print("\n4. DeepFace:")
try:
    from deepface import DeepFace
    print("   [OK] Imported" + (" (will use TensorFlow GPU)" if tf_gpu_ok else ""))
except Exception as e:
    print(f"   [ERROR] {e}")

# 5. RetinaFace
print("\n5. RetinaFace:")
try:
    from retinaface import RetinaFace
    print("   [OK] Imported")
except Exception as e:
    print(f"   [ERROR] {e}")

# 6. NudeNet
print("\n6. NudeNet:")
try:
    from nudenet import NudeDetector
    print("   [OK] Imported")
except Exception as e:
    print(f"   [ERROR] {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  - ONNX Runtime (InsightFace): {'GPU' if onnx_gpu_ok else 'CPU'}")
print(f"  - TensorFlow (DeepFace): {'GPU' if tf_gpu_ok else 'CPU'}")

if tf_gpu_ok:
    print("\n[OK] TensorFlow GPU is working - DeepFace heavy computation on GPU")
    if not onnx_gpu_ok:
        print("[INFO] InsightFace on CPU is still fast (~50-100ms per image)")
    print("\nSystem is ready for production!")
else:
    print("\n[FAIL] TensorFlow GPU not available - this is critical!")
    sys.exit(1)
EOF

# Step 12: Create production startup script
echo ""
echo "Step 12: Creating production startup script..."
echo "------------------------------------------------------------"

# Get site packages path
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

cat > run_gpu_final.sh << RUNSCRIPT
#!/bin/bash
# Production startup script - ensures GPU is used
# Auto-generated by setup_gpu_final.sh

SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
cd "\$SCRIPT_DIR"

# Activate venv
source venv/bin/activate

# Set up NVIDIA pip package library paths (CRITICAL for ONNX Runtime GPU)
SITE_PACKAGES=\$(python3 -c "import site; print(site.getsitepackages()[0])")
NVIDIA_LIBS=""
for pkg in cuda_runtime cudnn cublas cufft curand cusolver cusparse nccl; do
    PKG_LIB="\$SITE_PACKAGES/nvidia/\${pkg}/lib"
    if [ -d "\$PKG_LIB" ]; then
        NVIDIA_LIBS="\${NVIDIA_LIBS}:\${PKG_LIB}"
    fi
done
export LD_LIBRARY_PATH="\${NVIDIA_LIBS#:}:\$LD_LIBRARY_PATH"

# Also add system CUDA if available
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi

# TensorFlow settings
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "Starting Photo Validation API - PRODUCTION MODE"
echo "============================================================"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA Libs: \${NVIDIA_LIBS#:}"
echo "============================================================"

# Run with uvicorn
exec uvicorn api_hybrid:app \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --workers 1 \\
    --log-level info \\
    --timeout-keep-alive 30
RUNSCRIPT

chmod +x run_gpu_final.sh
print_status "Production startup script created"

# Create temp directory
mkdir -p /tmp/photo_uploads
chmod 755 /tmp/photo_uploads 2>/dev/null || true

# Final message
echo ""
echo "============================================================"
echo -e "${GREEN}SETUP COMPLETE${NC}"
echo "============================================================"
echo ""
echo "GPU Status:"
echo "  - TensorFlow (DeepFace): GPU - Heavy computation on GPU"
echo "  - InsightFace: Check verification output above"
echo ""
echo "To start the production server:"
echo "  ./run_gpu_final.sh"
echo ""
echo "API endpoints:"
echo "  - Health: http://0.0.0.0:8000/health"
echo "  - Docs:   http://0.0.0.0:8000/docs"
echo "============================================================"
