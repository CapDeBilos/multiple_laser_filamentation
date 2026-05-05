#!/bin/bash
set -euo pipefail

export CUDA_HOME=/usr/local/cuda-12.6.3
export CUDA_PATH=/usr/local/cuda-12.6.3
export CPATH="/usr/local/cuda-12.6.3/targets/x86_64-linux/include:${CPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6.3/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda-12.6.3/bin:${PATH:-}"

pin=${1:?Usage: ./square_4D_FFT_worker.sh PIN_FACTOR}

echo "CUDA_HOME=$CUDA_HOME"
echo "PATH=$PATH"
echo "Checking CUDA..."
which nvcc || true
nvcc --version || true
python -c "import cupy as cp; x=cp.ones((10,10)); print('CuPy OK:', cp.sum(x).item())"

echo "Running Square 4D FFT artificial-plasma simulation at Pin/Pcr=$pin"
python square_4D_FFT_artificial_plasma_simulation.py "$pin" "$pin" --step 0.5
echo "Finished Square Pin/Pcr=$pin"
