FROM python:3.12-slim

# Prevent Python from buffering logs and creating .pyc files
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
# Application
# --------------------------------------------------

COPY app ./app
COPY scripts ./scripts

# The trained model is intentionally NOT copied.
# Mount checkpoints/grayscale/best_model at runtime.

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]