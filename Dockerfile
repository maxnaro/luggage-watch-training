FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy source code
COPY source/ /app/source/

# Mount points (data, runs, model, config)
VOLUME ["/app/data", "/app/runs", "/app/model"]

# Unified entrypoint — pass "train" or "export" as the command
ENTRYPOINT ["python3", "source/entrypoint.py"]

# Default to training
CMD ["train"]