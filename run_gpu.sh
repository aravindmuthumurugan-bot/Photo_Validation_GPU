#!/bin/bash
# ============================================================
# Run Script for Photo Validation API on GPU
# ============================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================"
echo "Starting Photo Validation API (GPU Mode)"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running setup...${NC}"
    ./setup_gpu.sh
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variables for optimal GPU performance
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

# Optional: Set CUDA paths if needed
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo ""
echo "Environment:"
echo "  - Python: $(python --version)"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  - Working directory: $SCRIPT_DIR"
echo ""

# Check GPU availability before starting
python3 << 'EOF'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Status: {len(gpus)} GPU(s) available")
    else:
        print("GPU Status: Running on CPU")
except:
    pass

try:
    import onnxruntime as ort
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("ONNX Runtime: CUDA enabled")
    else:
        print("ONNX Runtime: CPU only")
except:
    pass
EOF

echo ""
echo "Starting server..."
echo "============================================================"

# Run with uvicorn
# Options:
#   --host 0.0.0.0   : Listen on all interfaces
#   --port 8000      : Port number
#   --workers 1      : Single worker (important for GPU memory)
#   --reload         : Auto-reload on code changes (remove in production)

exec uvicorn api_hybrid:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
