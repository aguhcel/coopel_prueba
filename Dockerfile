FROM python:3.11.13-slim-bullseye

# Set environment variables for development
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/usr/src/coopel_prueba

# Create working directory
WORKDIR /usr/src/coopel_prueba

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY 03_requirements_docker.txt .
RUN pip install --upgrade pip && \
    pip install -r 03_requirements_docker.txt
#    pip install "fastapi[standard]" && \
#    pip install pycaret


# Copy source code (this will be overridden by volume mount)
COPY . .

# Create necessary directories
RUN mkdir -p data models logs

# Expose port
EXPOSE 8000

# Use CMD instead of ENTRYPOINT for flexibility
CMD ["uvicorn", "app.api.router.v0.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]