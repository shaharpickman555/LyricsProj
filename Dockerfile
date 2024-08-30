FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
USER root
ENV TORCH_HOME=/data/models
WORKDIR /app

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    streamlit \
    yt-dlp \
    faster-whisper \
    demucs \
    ffmpeg-python
RUN git clone --depth 1 --branch v4.0.1 --single-branch https://github.com/facebookresearch/demucs /lib/demucs
COPY cmdb.py .
RUN mkdir ./uploads
EXPOSE 8502
CMD ["streamlit", "run", "cmdb.py", "--server.port=8502", "--server.headless=true"]