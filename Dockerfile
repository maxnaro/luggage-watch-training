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

# Copy everything inside the 'training' submodule to /app
COPY . /app

# Dataset mounting point
VOLUME /app/data

# Entry point
ENTRYPOINT ["python3", "source/train.py"]

# Default arguments
CMD ["--data", "config/data.yaml", "--model", "yolo11n.pt", "--imgsz", "640", "--epochs", "100", "--batch", "16", "--project", "runs", "--name", "exp", "--device", ""]