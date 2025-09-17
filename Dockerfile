FROM python:3.10-slim

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir runpod

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler files
COPY rp_handler.py /
COPY qwen_edit_service.py /
COPY qwen_gen_service.py /
COPY shared_utils.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]
