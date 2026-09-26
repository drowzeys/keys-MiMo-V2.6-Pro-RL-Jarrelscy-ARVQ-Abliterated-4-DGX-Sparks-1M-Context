#!/bin/bash
# One rank of the four-Spark Abliterated champion (image v3, 2026-09-26).
#   launch-rank.sh <this-node-ip> <rank 0-3> <roce-gid-index> <host-checkpoint-path> [api|headless]
#
# HOSTPATH should be the Abliterated tree (…-ablit-dealign-op) or the gated HF download
# drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated.
#
# Champion defaults (override with env):
#   MAXLEN=1048576 SEQS=4 BATCHED=5120 LM_ONLY=1
#   SPEC='{"method":"mtp","num_speculative_tokens":2}'   # all three MTP heads, non-chain
#   COMPILE=<torch.compile + FULL_AND_PIECEWISE CUDA graphs>; COMPILE=eager to disable
# Always: --enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo and
# a default output cap of 8192 tokens (the checkpoint's generation_config said 2048).
#
# Rank 0 uses mode api. The other three use headless. Start ranks 1-3 first.
# GPU memory fraction is fixed at 0.85.
set -euo pipefail
IMAGE="${IMAGE:-ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:63430f7-sm121-v3}"
NAME="${NAME:-mimo26-arvq-tp4}"
PORT="${PORT:-8888}"
MASTER="${MASTER_ADDR:?set MASTER_ADDR to the rank-0 IP}"
MPORT="${MASTER_PORT:-29621}"
MODEL=/models/mimo-arvq
UTIL=0.85
MAXLEN="${MAXLEN:-1048576}"
HEAD_IP="$1"
RANK="$2"
GID="$3"
HOSTPATH="$4"
MODE="${5:-api}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

EXTRA=()
if [ "$MODE" = "headless" ]; then
  EXTRA+=(--headless)
fi
if [ "${LM_ONLY:-1}" = "1" ]; then
  EXTRA+=(--language-model-only)
fi
# SPEC=none disables speculative decoding.
DEFAULT_SPEC='{"method":"mtp","num_speculative_tokens":2}'
SPEC="${SPEC:-$DEFAULT_SPEC}"
if [ "$SPEC" != "none" ]; then
  EXTRA+=(--speculative-config "$SPEC")
fi
# torch.compile + CUDA graphs. Capture sizes cover MTP verify batches (k+1 per seq).
DEFAULT_COMPILE='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,6,8,9,12,16]}'
COMPILE="${COMPILE:-$DEFAULT_COMPILE}"
if [ "$COMPILE" = "eager" ]; then
  EXTRA+=(--enforce-eager)
else
  EXTRA+=(--compilation-config "$COMPILE")
fi

mkdir -p /var/tmp/mimo-arvq-vllm-cache   # persisted torch.compile cache

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME"
fi

exec docker run -d --name "$NAME" \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --network host --ipc host --shm-size 16g --gpus all --privileged \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOSTPATH:$MODEL:ro" \
  -v /var/tmp/mimo-arvq-vllm-cache:/root/.cache/vllm \
  --entrypoint /bin/bash \
  -e VLLM_HOST_IP="$HEAD_IP" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TOKENIZERS_PARALLELISM=false \
  -e NCCL_NET=IB \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
  -e GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
  -e TP_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
  -e NCCL_IB_GID_INDEX="$GID" \
  -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_NVLS_ENABLE=0 \
  -e PYTHONPATH=/opt/arvq/runtime \
  -e VLLM_PLUGINS= \
  "$IMAGE" \
  /opt/arvq/entrypoint.sh \
  serve "$MODEL" \
  --served-model-name MiMo-V2.6-Pro-ARVQ \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser mimo \
  --override-generation-config '{"max_new_tokens": 8192}' \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "${SEQS:-4}" \
  --max-num-batched-tokens "${BATCHED:-5120}" \
  --gpu-memory-utilization "$UTIL" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --nnodes 4 --node-rank "$RANK" \
  --master-addr "$MASTER" --master-port "$MPORT" \
  "${EXTRA[@]}"
