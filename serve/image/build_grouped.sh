#!/usr/bin/env bash
set -euo pipefail
d="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rm -f "$d/grouped.so"
"${NVCC:-nvcc}" -O3 -std=c++17 -gencode "arch=compute_${ARVQ_CUDA_ARCH:-121a},code=sm_${ARVQ_CUDA_ARCH:-121a}" \
  --shared -Xcompiler=-fPIC -Xptxas=-v "$d/grouped.cu" -o "$d/grouped.so" 2>&1 | grep -E "error|registers|spill|warning" || true
test -f "$d/grouped.so"
