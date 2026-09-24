#!/bin/bash
# One rank of the four-Spark champion.
#   launch-rank.sh <this-node-ip> <rank 0-3> <roce-gid-index> <host-checkpoint-path> [api|headless]
#
# Champion environment:
#   LM_ONLY=1 MAXLEN=1048576 SEQS=4 \
#   SPEC='{"method":"mtp","num_speculative_tokens":2}'
#
# Rank 0 uses mode api. The other three use headless.
# GPU memory fraction is fixed at 0.85.
set -euo pipefail
IMAGE="${IMAGE:-mimo26-arvq-spark:63430f7-sm121-v1}"
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
if [ -n "${SPEC:-}" ]; then
  EXTRA+=(--speculative-config "$SPEC")
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME"
fi

exec docker run -d --name "$NAME" \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --network host --ipc host --shm-size 16g --gpus all --privileged \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOSTPATH:$MODEL:ro" \
  -v "$ROOT/serve/entrypoint.sh:/opt/arvq/entrypoint.sh:ro" \
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
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "${SEQS:-4}" \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization "$UTIL" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enforce-eager \
  --nnodes 4 --node-rank "$RANK" \
  --master-addr "$MASTER" --master-port "$MPORT" \
  "${EXTRA[@]}"
