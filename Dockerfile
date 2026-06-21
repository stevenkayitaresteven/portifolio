# Container for the public Sentinel chat demo. Tuned to boot reliably on a free
# CPU Space: CPU-only torch, a single small text model (toxic-bert) baked in and
# served offline so the first message is instant and startup is fast. The image
# ViT and Whisper are left off here (the offline NudeNet covers images); flip
# SENTINEL_ENABLE_HF_IMAGE=1 / SENTINEL_ENABLE_HF_AUDIO=1 on a bigger box.
FROM python:3.11-slim

# opencv-python-headless needs libGL/libglib at runtime even on headless boxes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, from the CPU wheel index — keeps the image small and
# avoids downloading multi-GB CUDA libraries that a CPU Space can't use.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY . .

# detectors = offline NudeNet, hf = transformers (text model), serve = chat UI.
RUN pip install --no-cache-dir -e ".[detectors,hf,serve]"

# Enable the HF *text* model only, and use the single lightweight toxic-bert
# (catches the implicit toxicity/hate the word list misses).
ENV SENTINEL_ENABLE_HF=1
ENV SAFETY_MM_TEXT_MODELS=unitary/toxic-bert

# Bake just that one model into the image and serve it fully offline at runtime,
# so startup never waits on (or fails on) a network download. Cache must be
# writable — Spaces runs the container as a non-root user.
ENV HF_HOME=/app/.hf_cache
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
RUN mkdir -p /app/.hf_cache && \
    python -c "from transformers import pipeline; pipeline('text-classification', model='unitary/toxic-bert', top_k=None, truncation=True)" && \
    chmod -R 777 /app/.hf_cache
# Use the baked cache, never hit the network on model load at runtime.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Hugging Face Spaces routes to 7860; other hosts inject $PORT (handled in app.py).
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
