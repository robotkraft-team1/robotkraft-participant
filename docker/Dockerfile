# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12.9
FROM python:${PYTHON_VERSION}-slim

ARG UV_VERSION=0.11.21
# cpu : sans la pile CUDA, environ trois fois plus legere. cuda : pour le
# serveur d'entrainement et les portables equipes d'un GPU NVIDIA.
ARG VARIANTE=cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/workspace/.venv/bin:$PATH \
    HF_HOME=/workspace/.cache/huggingface \
    HF_LEROBOT_HOME=/workspace/.cache/huggingface/lerobot \
    TORCH_HOME=/workspace/.cache/torch \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy

# Couche systeme : ne change quasiment jamais, donc jamais reconstruite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 ffmpeg \
    libusb-1.0-0 \
    libgl1 \
    libsm6 libice6 \
    fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /workspace

# Dependances. Le cache BuildKit survit entre deux constructions : une modification
# de uv.lock ne retelecharge que les paquets qui ont reellement change, torch et les
# autres poids lourds sont repris du cache.
#
# VARIANTE=kinematics : la base cpu plus le solveur IK (placo). On n'ajoute pas de
# torch CUDA, l'IK ne demande que du CPU ; d'ou "base cpu + extra kinematics".
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/opt/uv-cache,sharing=locked \
    uv venv .venv && \
    case "${VARIANTE}" in \
      kinematics) uv sync --frozen --no-dev --extra cpu --extra kinematics ;; \
      *)         uv sync --frozen --no-dev --extra "${VARIANTE}" ;; \
    esac

RUN mkdir -p .venv/lib/python3.12/site-packages/cv2/qt/fonts \
    && cp /usr/share/fonts/truetype/dejavu/*.ttf .venv/lib/python3.12/site-packages/cv2/qt/fonts/ 2>/dev/null || true

# En dernier : ce qui change le plus souvent, pour ne rien invalider en amont
COPY scripts/ scripts/
# L'URDF calibré d'usine du SO-101, utilisé par la cinématique (scripts/kinematics/).
COPY urdf/ urdf/

CMD ["/bin/bash"]
