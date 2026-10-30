# Serving image for NVIDIA-Nemotron-3.5-Transcribe-30B-A4B.
#
# Replicates the model card's Environment Setup verbatim: vLLM 0.32.0 and the
# NeMo Speech main branch (which provides the nemo_speechlm vLLM plugin) in one
# uv-locked Python 3.13 environment, plus PyAV (compressed-audio decode) and
# OpenCV (a vLLM requirement NeMo excludes). The uv.lock pins
# torch 2.11.0+cu129 and flashinfer-python 0.7.0.post1, both Blackwell-ready.
FROM ubuntu:22.04

# Pinned NeMo Speech revision; a new ref is a new baseline.
ARG NEMO_SPEECH_REF=548fd8665b8e0cb7ad0d1f58adfec7489fda1d09

# uv comes from PyPI: ghcr.io is not reachable from this host.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git curl python3 python3-pip \
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

ENV PATH="/opt/nemo-speech/.venv/bin:${PATH}" \
    VLLM_PLUGINS=nemo_speechlm \
    VLLM_MAX_AUDIO_DECODE_DURATION_S=3600 \
    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180 \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/tmp

ENTRYPOINT ["/opt/nemo-speech/.venv/bin/vllm"]
