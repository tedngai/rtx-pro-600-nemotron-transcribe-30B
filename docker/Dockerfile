# Serving image for NVIDIA-Nemotron-3.5-Transcribe-30B-A4B.
#
# Replicates the model card's Environment Setup verbatim: vLLM 0.32.0 and the
# NeMo Speech main branch (which provides the nemo_speechlm vLLM plugin) in one
# uv-locked Python 3.13 environment, plus PyAV (compressed-audio decode) and
# OpenCV (a vLLM requirement NeMo excludes). The uv.lock resolves torch
# 2.13.0+cu130 for py3.13/linux and flashinfer-python 0.7.0.post1.
#
# The runtime needs a C/C++ toolchain: Triton compiles its CUDA driver module
# at import time, torch.compile/Inductor builds flex_attention kernels, and
# FlashInfer JIT-compiles its kernels unless precompiled cubins exist.
FROM ubuntu:22.04

# Pinned NeMo Speech revision; a new ref is a new baseline.
ARG NEMO_SPEECH_REF=548fd8665b8e0cb7ad0d1f58adfec7489fda1d09

# uv comes from PyPI: ghcr.io is not reachable from this host.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git curl python3 python3-pip build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --no-cache-dir uv

RUN git clone https://github.com/NVIDIA-NeMo/Speech.git /opt/nemo-speech \
 && git -C /opt/nemo-speech checkout "${NEMO_SPEECH_REF}"

# uv-managed interpreters must live outside /root (0700 in ubuntu:22.04): the
# server runs as a non-root uid and could not traverse to the venv's stdlib.
# UV_PYTHON overrides the repo's .python-version (3.12) to keep the card's
# recipe: uv venv --python 3.13.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON=3.13

WORKDIR /opt/nemo-speech
RUN uv venv --python 3.13 .venv \
 && UV_PROJECT_ENVIRONMENT=/opt/nemo-speech/.venv uv sync --extra asr --extra vllm --locked --no-install-project \
 && uv pip install --python .venv/bin/python --no-deps -e . \
 && uv pip install --python .venv/bin/python --no-deps av==19.0.2 opencv-python-headless==5.0.0.93

# Precompiled FlashInfer kernels matching flashinfer-python 0.7.0.post1+cu130.
# torch 2.13 bundles nvcc 13.4 with CUDA 13.0 headers, a combination that
# FlashInfer's CCCL compatibility check rejects; precompiled cubins skip JIT.
RUN uv pip install --python /opt/nemo-speech/.venv/bin/python --upgrade \
      --index-url https://flashinfer.ai/whl/cu130 \
      flashinfer-jit-cache==0.7.0.post1+cu130

# FlashInfer JIT-compiles some kernels on demand (e.g. the mamba SSU ops the
# card's --mamba-backend flashinfer needs). torch's bundled cu13 toolkit pairs
# nvcc 13.4 with CUDA 13.0 headers, which FlashInfer's CCCL check rejects, so
# install a matching CUDA 13.0 compiler from NVIDIA's apt repository.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget gnupg2 \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-nvcc-13-0 cuda-cudart-dev-13-0 \
 && rm -f cuda-keyring_1.1-1_all.deb \
 && rm -rf /var/lib/apt/lists/*

# torch 2.13's cu130 wheels bundle a complete CUDA 13 toolkit (nvcc, headers,
# libcudart) under nvidia/cu13; point CUDA_HOME at it so FlashInfer's JIT can
# compile kernels. ninja ships in the venv.
ENV PATH="/usr/local/cuda-13.0/bin:/opt/nemo-speech/.venv/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda-13.0" \
    VLLM_PLUGINS=nemo_speechlm \
    VLLM_MAX_AUDIO_DECODE_DURATION_S=3600 \
    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180 \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/tmp

ENTRYPOINT ["/opt/nemo-speech/.venv/bin/vllm"]
