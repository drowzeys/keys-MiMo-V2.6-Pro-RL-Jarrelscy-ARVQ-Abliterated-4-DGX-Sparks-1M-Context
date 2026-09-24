# Thin layer on the SM121 image built from Jarrelscy's sm120 fork.
# The parent image compiles the ARVQ kernels. This layer adds the Spark
# loader fixes that were bind-mounted during the first successful serve.
FROM mimo26-arvq-spark:63430f7-sm121-v1

COPY mimo_v2.py /opt/arvq/runtime/vllm/model_executor/models/mimo_v2.py
COPY mimo_v2_mtp.py /opt/arvq/runtime/vllm/model_executor/models/mimo_v2_mtp.py
COPY mimo_v2_omni.py /opt/arvq/runtime/vllm/model_executor/models/mimo_v2_omni.py
COPY nvfp4_arvq_hybrid.py /opt/arvq/runtime/vllm/model_executor/layers/quantization/nvfp4_arvq_hybrid.py
COPY act_quant_fusion.py /opt/arvq/runtime/vllm/compilation/passes/fusion/act_quant_fusion.py
COPY qwen3_dflash.py /opt/arvq/runtime/vllm/model_executor/models/qwen3_dflash.py
COPY entrypoint.sh /opt/arvq/entrypoint.sh

RUN chmod 755 /opt/arvq/entrypoint.sh
ENTRYPOINT ["/opt/arvq/entrypoint.sh"]
