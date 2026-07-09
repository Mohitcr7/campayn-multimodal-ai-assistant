# ─────────────────────────────────────────────────────────────────
# Dockerfile
# Campayn AI Assistant — Container Image
#
# WHAT THIS DOES
# ---------------
# Packages the entire app into a reproducible container image.
# The same image runs on any machine — your laptop, a GCP VM,
# or a Kubernetes cluster — without "it works on my machine" issues.
#
# BUILD & RUN
# ------------
#   docker build -t campayn-ai .
#   docker run -p 8000:8000 --env-file .env campayn-ai
#
# HOW DOCKER LAYERS WORK (for beginners)
# ----------------------------------------
# Each instruction creates a LAYER — a snapshot of the filesystem.
# Layers are cached. If requirements.txt hasn't changed, Docker
# reuses the cached pip install layer — rebuilds are fast.
# Order matters: put rarely-changing things first (base image, system
# deps, requirements) and frequently-changing things last (your code).
# ─────────────────────────────────────────────────────────────────

# ── Base image ─────────────────────────────────────────────────
# python:3.11-slim = Python 3.11 on minimal Debian (no dev tools,
# no docs, no test files). ~130MB vs ~900MB for the full image.
FROM python:3.11-slim

# ── Working directory ──────────────────────────────────────────
# All subsequent commands run from /app inside the container.
# Files we COPY go here. Processes we RUN start from here.
WORKDIR /app

# ── System dependencies ────────────────────────────────────────
# We need build-essential for packages that compile C extensions
# (chromadb uses Rust/C under the hood).
# --no-install-recommends keeps the image lean.
# Clean apt cache afterwards to reduce image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────
# Copy requirements FIRST (before the rest of the code).
# WHY: Docker caches each layer. If we copied all code first,
# any code change would bust the pip install cache and
# reinstall everything — slow. This way, pip only reruns
# when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────
# Copy source files. This layer is rebuilt on every code change,
# but that's fine — code copying is instant.
COPY . .

# ── Data directories ───────────────────────────────────────────
# Create directories that will hold data at runtime.
# These are empty in the image — mounted via Docker volumes
# or populated by the startup scripts.
RUN mkdir -p data/raw data/processed data/vector_store logs

# ── Non-root user ──────────────────────────────────────────────
# Running as root inside a container is a security risk.
# Create a dedicated user and switch to it.
# This is a production security best practice.
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Port ───────────────────────────────────────────────────────
# Tell Docker which port this container listens on.
# EXPOSE is documentation — the actual binding happens in
# docker-compose.yml with "ports: - 8000:8000"
EXPOSE 8000

# ── Default command ────────────────────────────────────────────
# What runs when you do `docker run campayn-ai`.
# --host 0.0.0.0  = listen on all interfaces (not just localhost)
#                   Required inside a container — otherwise the
#                   app only listens inside the container and
#                   external traffic never reaches it.
# --workers 1     = single worker (stateful LangGraph checkpointer
#                   doesn't support multiple workers without Redis)
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]