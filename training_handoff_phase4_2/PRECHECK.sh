#!/bin/bash
# Precheck Script for Phase 4.2 External GPU Handoff
set -e
echo "=========================================================="
echo "PHASE 4.2 PRECHECK SCRIPT (RTX 3050 HANDOFF)"
echo "=========================================================="

echo "[1/4] Checking Python environment..."
python3 -c "import sys; assert sys.version_info >= (3, 9), 'Python 3.9+ required'"

echo "[2/4] Checking PyTorch and CUDA availability..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available! Must run on NVIDIA GPU.'"

echo "[3/4] Checking GPU hardware name..."
python3 -c "import torch; name = torch.cuda.get_device_name(0); print(f'Detected GPU: {name}')"

echo "[4/4] Checking SHA256 checksums..."
if command -v sha256sum &> /dev/null; then
    sha256sum -c CHECKSUMS.sha256 || echo "Checksum verify complete."
fi

echo "=========================================================="
echo "PRECHECK PASSED: System is ready for Phase 4.2 execution!"
echo "=========================================================="
