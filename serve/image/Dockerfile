# ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:63430f7-sm121-v3 (also :latest)
# The only supported image. Pull it; rebuild only if you must.
#
# On top of the ARVQ base (Dockerfile.base):
#   mimo_v2.py            Spark loader fixes; ablit capture hooks gated at import
#   mimo_v2_mtp.py        all three MTP heads, non-chain multi-layer propose
#   speculative.py        routes multi-layer MiMo MTP to the per-step proposer
#   chat_serving.py       keeps finish_reason "length" on truncated tool batches
#   nvfp4_arvq_hybrid.py  + nvfp4_arvq_batched_prefill.py + grouped.cu:
#                         expert-batched ARVQ prefill (2-2.6x faster prefill)
#   mimo_v2_omni.py, nvfp4 loader, act_quant_fusion.py, qwen3_dflash.py: port fixes
FROM mimo26-arvq-base:63430f7-sm121

ARG Q=/opt/arvq/runtime/vllm/model_executor/layers/quantization
ARG M=/opt/arvq/runtime/vllm/model_executor/models
COPY grouped.cu build_grouped.sh /tmp/grouped/
RUN NVCC=/usr/local/cuda/bin/nvcc ARVQ_CUDA_ARCH=121a bash /tmp/grouped/build_grouped.sh \
    && cp /tmp/grouped/grouped.so $Q/arvq/grouped.so && rm -rf /tmp/grouped
COPY mimo_v2.py mimo_v2_mtp.py mimo_v2_omni.py qwen3_dflash.py $M/
COPY nvfp4_arvq_hybrid.py nvfp4_arvq_batched_prefill.py $Q/
COPY act_quant_fusion.py /opt/arvq/runtime/vllm/compilation/passes/fusion/act_quant_fusion.py
COPY speculative.py /opt/arvq/runtime/vllm/config/speculative.py
COPY chat_serving.py /opt/arvq/runtime/vllm/entrypoints/openai/chat_completion/serving.py
COPY entrypoint.sh /opt/arvq/entrypoint.sh

ENV VLLM_ARVQ_GROUPED_PREFILL=1 \
    VLLM_ARVQ_COMPACT_PREFILL=1 \
    VLLM_ARVQ_SORT_NATIVE_PREFILL=1 \
    VLLM_ARVQ_FUSED_ACTIVATION_PACK=1 \
    VLLM_ARVQ_BATCHED_PREFILL=1 \
    MIMO_MTP_LAYERS=3 \
    MIMO_ABLIT_CAPTURE=0

RUN chmod 755 /opt/arvq/entrypoint.sh
ENTRYPOINT ["/opt/arvq/entrypoint.sh"]
