#!/bin/bash
set -e

export CUDA_HOME=/usr/local/cuda-12.6.3
export CUDA_PATH=/usr/local/cuda-12.6.3
export CPATH="/usr/local/cuda-12.6.3/targets/x86_64-linux/include:${CPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6.3/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda-12.6.3/bin:${PATH:-}"

start=$1
end=$2

echo "CUDA_HOME=$CUDA_HOME"
echo "PATH=$PATH"
echo "Checking CUDA..."
which nvcc || true
nvcc --version || true

python -c "import cupy as cp; x=cp.ones((10,10)); print('CuPy OK:', cp.sum(x).item())"

echo "Running FFT sweep from $start to $end"

python THOMAS_PCR_adaptive_FFT_sweep_xy_no_t_memory_optimized.py "$start" "$end"

echo "Finished sweep $start to $end"