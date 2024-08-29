FROM pytorch/pytorch:latest
USER root
ENV TORCH_HOME=/data/models
ENV OMP_NUM_THREADS=1
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
RUN git clone --depth 1 --branch v4.0.0 --single-branch https://github.com/facebookresearch/demucs /lib/demucs
COPY cmdb.py .
RUN mkdir ./uploads
EXPOSE 8502
CMD ["streamlit", "run", "cmdb.py", "--server.port=8502", "--server.headless=true"]

#FROM python:3.9-slim
#WORKDIR /app
#RUN apt-get update && apt-get install -y \
#    ffmpeg \
#    && rm -rf /var/lib/apt/lists/*
#RUN pip install --no-cache-dir \
#    streamlit \
#    yt-dlp \
#    faster-whisper \
#    demucs \
#    ffmpeg-python
#RUN pip show demucs
#COPY cmdb.py .
#RUN mkdir ./uploads
#EXPOSE 8502
#CMD ["streamlit", "run", "cmdb.py", "--server.port=8502", "--server.headless=true"]

#FROM python:3.9
#WORKDIR /app
#COPY ./cmdb.py ./requirements.txt /app
#RUN pip install --upgrade pip
#RUN pip install --no-cache-dir -r requirements.txt
#CMD ["streamlit", "run", "./cmdb.py"]