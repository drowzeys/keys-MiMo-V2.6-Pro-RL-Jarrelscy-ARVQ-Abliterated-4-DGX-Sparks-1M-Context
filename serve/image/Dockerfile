# v2: thin layer on the SM121 image built from Jarrelscy's sm120 fork.
# The parent image compiles the ARVQ kernels. This layer adds the Spark loader
# fixes plus the 2026-09-26 speed and tool-call fixes:
#   mimo_v2.py        ablit capture hooks gated at import (torch.compile-clean)
#   mimo_v2_mtp.py    all three MTP heads, non-chain multi-layer propose
#   speculative.py    routes multi-layer MiMo MTP to the per-step proposer
#   chat_serving.py   keeps finish_reason "length" on truncated tool-call batches
FROM mimo26-arvq-spark:63430f7-sm121-v1

COPY mimo_v2.py /opt/arvq/runtime/vllm/model_executor/models/mimo_v2.py
COPY mimo_v2_mtp.py /opt/arvq/runtime/vllm/model_executor/models/mimo_v2_mtp.py
COPY mimo_v2_omni.py /opt/arvq/runtime/vllm/model_executor/models/mimo_v2_omni.py
COPY nvfp4_arvq_hybrid.py /opt/arvq/runtime/vllm/model_executor/layers/quantization/nvfp4_arvq_hybrid.py
COPY act_quant_fusion.py /opt/arvq/runtime/vllm/compilation/passes/fusion/act_quant_fusion.py
COPY qwen3_dflash.py /opt/arvq/runtime/vllm/model_executor/models/qwen3_dflash.py
COPY speculative.py /opt/arvq/runtime/vllm/config/speculative.py
COPY chat_serving.py /opt/arvq/runtime/vllm/entrypoints/openai/chat_completion/serving.py
COPY entrypoint.sh /opt/arvq/entrypoint.sh

# Champion defaults: ARVQ grouped/compact/sorted prefill, fused decode
# activation pack, all three MTP heads. Override with -e at docker run.
ENV VLLM_ARVQ_GROUPED_PREFILL=1 \
    VLLM_ARVQ_COMPACT_PREFILL=1 \
    VLLM_ARVQ_SORT_NATIVE_PREFILL=1 \
    VLLM_ARVQ_FUSED_ACTIVATION_PACK=1 \
    MIMO_MTP_LAYERS=3 \
    MIMO_ABLIT_CAPTURE=0

RUN chmod 755 /opt/arvq/entrypoint.sh
ENTRYPOINT ["/opt/arvq/entrypoint.sh"]
