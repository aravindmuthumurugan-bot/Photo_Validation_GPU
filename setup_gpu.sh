#!/bin/bash
# ============================================================
# GPU Setup Script for Photo Validation API
# Target: NVIDIA L40S with CUDA 12.4, Python 3.12.3
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "Photo Validation GPU Setup Script"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as the correct user
echo ""
echo "Step 1: Checking environment..."
echo "------------------------------------------------------------"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
print_status "Python version: $PYTHON_VERSION"

# Check NVIDIA driver
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Unknown")
    print_status "GPU detected: $GPU_INFO"
else
    print_error "nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_status "CUDA version: $CUDA_VERSION"
else
    print_warning "nvcc not found. CUDA toolkit may not be in PATH."
fi

# Step 2: Create virtual environment
echo ""
echo "Step 2: Creating virtual environment..."
echo "------------------------------------------------------------"

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists. Removing..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
print_status "Virtual environment created at ./$VENV_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"
print_status "Virtual environment activated"

# Upgrade pip
echo ""
echo "Step 3: Upgrading pip..."
echo "------------------------------------------------------------"
pip install --upgrade pip setuptools wheel
print_status "pip upgraded"

# Step 4: Install PyTorch with CUDA support (needed for some dependencies)
echo ""
echo "Step 4: Installing PyTorch with CUDA 12.4 support..."
echo "------------------------------------------------------------"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
print_status "PyTorch installed with CUDA 12.4"

# Step 5: Install ONNX Runtime GPU
echo ""
echo "Step 5: Installing ONNX Runtime GPU..."
echo "------------------------------------------------------------"
pip install onnxruntime-gpu==1.20.1
print_status "ONNX Runtime GPU installed"

# Step 6: Install TensorFlow with CUDA
echo ""
echo "Step 6: Installing TensorFlow..."
echo "------------------------------------------------------------"
pip install tensorflow[and-cuda]==2.18.0
print_status "TensorFlow installed"

# Step 7: Install remaining requirements
echo ""
echo "Step 7: Installing remaining dependencies..."
echo "------------------------------------------------------------"

# Install core dependencies
pip install \
    insightface==0.7.3 \
    deepface==0.0.93 \
    retina-face==0.0.17 \
    mtcnn==1.0.0 \
    nudenet==3.4.2

print_status "Face analysis libraries installed"

pip install \
    opencv-python-headless==4.10.0.84 \
    pillow==11.0.0 \
    scikit-image==0.24.0 \
    albumentations==1.4.21

print_status "Image processing libraries installed"

pip install \
    fastapi==0.115.6 \
    "uvicorn[standard]==0.34.0" \
    python-multipart==0.0.20 \
    pydantic==2.10.3

print_status "Web framework installed"

pip install \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.14.1 \
    requests==2.32.3 \
    tqdm==4.67.1 \
    gdown==5.2.0 \
    PyYAML==6.0.2 \
    python-dotenv==1.0.1

print_status "Utility libraries installed"

# Step 8: Download InsightFace models
echo ""
echo "Step 8: Downloading InsightFace models..."
echo "------------------------------------------------------------"

# Create models directory
mkdir -p ~/.insightface/models

# Download buffalo_l model if not present
if [ ! -d ~/.insightface/models/buffalo_l ]; then
    print_status "Downloading InsightFace buffalo_l model..."
    python3 -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('Model downloaded successfully')
"
else
    print_status "InsightFace models already present"
fi

# Step 9: Verify GPU access
echo ""
echo "Step 9: Verifying GPU access..."
echo "------------------------------------------------------------"

python3 << 'EOF'
import sys

print("Checking TensorFlow GPU...")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  TensorFlow GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f"    - {gpu}")
    else:
        print("  WARNING: No TensorFlow GPU found")
except Exception as e:
    print(f"  TensorFlow error: {e}")

print("\nChecking ONNX Runtime GPU...")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  Available providers: {providers}")
    if 'CUDAExecutionProvider' in providers:
        print("  CUDA provider available!")
    else:
        print("  WARNING: CUDA provider not available")
except Exception as e:
    print(f"  ONNX Runtime error: {e}")

print("\nChecking PyTorch GPU...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: PyTorch CUDA not available")
except Exception as e:
    print(f"  PyTorch error: {e}")
EOF

# Step 10: Create upload directory
echo ""
echo "Step 10: Creating upload directory..."
echo "------------------------------------------------------------"
mkdir -p /tmp/photo_uploads
chmod 755 /tmp/photo_uploads
print_status "Upload directory created at /tmp/photo_uploads"

# Final summary
echo ""
echo "============================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the API server:"
echo "  ./run_gpu.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python api_hybrid.py"
echo ""
echo "API will be available at: http://0.0.0.0:8000"
echo "Health check: http://0.0.0.0:8000/health"
echo "API docs: http://0.0.0.0:8000/docs"
echo "============================================================"
