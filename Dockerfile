########################################
# SmartPark on Jetson Nano using YOLOv8
########################################

# ────────────────────────────────────────────────────────────────
# Base: L4T PyTorch container with Python 3.8+, PyTorch, CUDA/cuDNN
# (choose the tag matching your JetPack; below is an example)
# ────────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# ────────────────────────────────────────────────────────────────
# 1) Strip out any extra apt sources (incl. Kitware) so update never fails
# ────────────────────────────────────────────────────────────────
RUN rm -rf /etc/apt/sources.list.d && \
    sed -i '/kitware/d' /etc/apt/sources.list

# ────────────────────────────────────────────────────────────────
# 2) Install system dependencies
# ────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip \
    #   python3-opencv \
      python3-dev \
      libgl1 \
      curl \
      git \
      unzip \
      ffmpeg \
      nano && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────────────────────
# 3) Copy your application code into the container
# ────────────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app

# ────────────────────────────────────────────────────────────────
# 4) Install Python dependencies, including ultralytics (YOLOv8)
#    requirements.txt must list:
#      ultralytics>=8.0.0
#      opencv-python-headless
#      numpy
# ────────────────────────────────────────────────────────────────
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# ────────────────────────────────────────────────────────────────
# 5) Default command: run your video detector
# ────────────────────────────────────────────────────────────────
CMD ["python3","parking_detector_video.py","carPark.mp4","--spots","spots_video.json"]

