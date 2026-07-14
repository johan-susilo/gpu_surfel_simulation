# Use Debian Bookworm for a stable, predictable package environment
FROM python:3.11-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Removed libgl1-mesa-glx (libgl1 handles it natively now)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYVISTA_OFF_SCREEN=true

# Copy just the requirements file first to cache the heavy pip installs
COPY requirements.txt .

# Install your Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your actual codebase into the container
COPY core/ ./core/
COPY params/ ./params/
COPY main.py .

# Use xvfb-run to wrap the execution, giving it a virtual display
CMD ["xvfb-run", "-a", "python", "main.py", "d0"]