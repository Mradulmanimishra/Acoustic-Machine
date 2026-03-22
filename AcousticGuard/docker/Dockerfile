# ── AcousticGuard Production Dockerfile ──────────────────────────────────────
# Multi-stage build: keeps the final image lean by separating build-time
# dependencies (compiler, pip cache) from runtime dependencies.
#
# Usage:
#   docker build -t acousticguard -f docker/Dockerfile .
#   docker run -p 8080:8080 -v /your/audio:/data acousticguard
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system libs needed to compile audio/torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into a clean prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# Runtime audio libs only (no compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Create a non-root user — best practice for production containers
RUN useradd --create-home --shell /bin/bash acousticguard
USER acousticguard
WORKDIR /app

# Copy application source
COPY --chown=acousticguard:acousticguard src/       ./src/
COPY --chown=acousticguard:acousticguard configs/   ./configs/
COPY --chown=acousticguard:acousticguard demo/      ./demo/

# If a pre-trained model is available, include it
# COPY --chown=acousticguard:acousticguard checkpoints/ ./checkpoints/

# Mount point for user audio data
VOLUME ["/data"]

# Expose port for future REST API / dashboard
EXPOSE 8080

# Default command: run inference on anything in /data
CMD ["python", "src/inference.py", "--normal_dir", "/data/normal", "--anomaly_dir", "/data/anomaly"]
