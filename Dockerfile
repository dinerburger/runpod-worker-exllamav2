# Base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

ARG MODEL_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG MODEL_REVISION=""
ARG KV_CACHE_QUANT="FP16"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    BASE_PATH=$BASE_PATH \
    KV_CACHE_QUANT=$KV_CACHE_QUANT \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 

ENV PYTHONPATH="/:/exllama2-workspace"

COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
