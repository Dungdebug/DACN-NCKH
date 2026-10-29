# ENVIRONMENT SPECIFICATION — PHASE 4.2
## Hardware Target: NVIDIA GeForce RTX 3050 (6GB / 8GB)

### 1. Python Environment
- Python Version: `>= 3.9.10, <= 3.11.x`
- PyTorch: `>= 2.1.0` with CUDA `cu118` or `cu121`
- Transformers: `>= 4.38.0`
- Datasets: `>= 2.18.0`
- Accelerate: `>= 0.27.0`
- NumPy: `>= 1.24.0`
- SciPy: `>= 1.10.0`

### 2. CUDA & Memory Settings
- CUDA Device: `cuda:0`
- Target Precision: `float32` (training) / `float16` (evaluation)
- Max Context Tokens: 512
- Peak Memory Limit: `< 3500 MB`
