# Lightweight container for the public Sentinel chat demo.
# Runs the offline moderation floor (lexicon + PII regex + file gate + NudeNet),
# which boots in seconds on a free CPU instance. To run the Hugging Face models
# too, add the `hf` extra below and set SENTINEL_ENABLE_HF=1.
FROM python:3.11-slim

# opencv-python-headless needs libGL/libglib at runtime even on headless boxes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# `detectors` = offline NudeNet, `serve` = FastAPI/uvicorn for the chat UI.
RUN pip install --no-cache-dir -e ".[detectors,serve]"

# Hugging Face Spaces routes to 7860; other hosts inject $PORT (handled in app.py).
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
