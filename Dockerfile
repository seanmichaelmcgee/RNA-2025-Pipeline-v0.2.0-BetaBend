# Dockerfile for RNA Folding Project - V1 PyTorch Pipeline
# Base image with CUDA 12.1.1 and cuDNN 8, matching environment.yml target
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set ARG for noninteractive installs
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

# Install essential system dependencies and build tools
# Clean up apt lists afterwards to keep image size down
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    # Add any other essential system libs here if needed by dependencies later
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda to /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    # Initialize conda for bash shells
    /opt/conda/bin/conda init bash && \
    # Perform cleanup
    /opt/conda/bin/conda clean -afy

# Add conda to PATH (for subsequent RUN commands)
ENV PATH /opt/conda/bin:$PATH

# Install mamba for faster environment solving
RUN mamba install -y -c conda-forge mamba

# --- Environment Creation ---
# Copy ONLY the environment file first to leverage Docker cache
COPY environment.yml /app/environment.yml
WORKDIR /app

# Create the conda environment using mamba
# This layer is cached and only re-runs if environment.yml changes
RUN mamba env create -f environment.yml

# --- Application Setup ---
# Set the default shell to use conda run, activating the environment
SHELL ["conda", "run", "-n", "rna-3d-folding", "/bin/bash", "-c"]

# Create directories for potential mount points (will be empty in image)
RUN mkdir -p /app/data /app/output /app/config /app/logs

# Copy the rest of the application code
# This is done AFTER environment creation to benefit from caching
# Ensure you have a .dockerignore file to exclude large files/dirs (like data/, venv/, .git/)
COPY . /app

# Verify environment activation (optional sanity check)
RUN echo "Checking Python and PyTorch versions in environment..." && \
    python --version && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}, version: {torch.version.cuda}')"

# Set default command (can be overridden when running the container)
# Example: docker run ... rna-3d:v1 python scripts/train.py --config config/default_config.yaml
CMD ["echo", "Environment 'rna-3d-folding' ready. Run your scripts, e.g., 'python scripts/train.py ...'"]
