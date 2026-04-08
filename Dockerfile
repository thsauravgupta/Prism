# ──────────────────────────────────────────────────────────────────────────────
#  SmartThings On-Device Predictive Model — Inference Container
#
#  Provides a containerised environment for running both predictive pipelines:
#    1. Two-Level Architecture  (XGBoost + LSTM + Heuristic fusion)
#    2. KG-GNN                  (Knowledge Graph GNN via TensorFlow)
#
#  Usage:
#    docker compose up              # runs the demo inference script
#    docker compose run inference python demo_inference.py --help
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="VIT_25ST07VIT Team"
LABEL description="SmartThings On-Device Predictive Model Inference"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements-inference.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy project source code
COPY Two_level_Arch/ ./Two_level_Arch/
COPY KG_GNN/ ./KG_GNN/
COPY demo_inference.py .
COPY README.md .

# Create output directory
RUN mkdir -p /app/output

# Expose any output directory as a volume
VOLUME ["/app/output"]

# Default: run the demo inference script
CMD ["python", "demo_inference.py", "--all"]
