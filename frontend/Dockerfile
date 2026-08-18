code DockerfileFROM python:3.12-slim

# --------------------------------------------------
# Python configuration
# --------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# --------------------------------------------------
# System dependencies
# --------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*


# --------------------------------------------------
# Python dependencies
# --------------------------------------------------

COPY requirements-docker.txt .

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    -r requirements-docker.txt


# --------------------------------------------------
# Download Hugging Face processor
# --------------------------------------------------

RUN python -c "\
from transformers import LayoutLMv3Processor; \
LayoutLMv3Processor.from_pretrained( \
    'microsoft/layoutlmv3-base', \
    apply_ocr=False \
)"


# --------------------------------------------------
# Application
# --------------------------------------------------

COPY app ./app
COPY scripts ./scripts


# The trained classifier checkpoint is intentionally
# NOT copied into the image.
# It is mounted at runtime through docker-compose.


# --------------------------------------------------
# Runtime
# --------------------------------------------------

EXPOSE 8000

CMD [
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]