FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
USER root
WORKDIR /app

ENV TORCH_HOME=/data/models
ENV TZ=Asia/Jerusalem
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y software-properties-common
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /opt/conda/bin/ffmpeg
RUN update-alternatives --install /usr/bin/python3 python3 /opt/conda/bin/python3 0
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0
RUN update-alternatives --install /usr/bin/pip pip /opt/conda/bin/pip 0

RUN pip install --no-cache-dir -U \
    pip \
    setuptools
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git
RUN pip install --no-deps socketio
RUN pip install --no-cache-dir -U \
    flask-socketio \
    websocket-client \
    requests \
    aiohttp \
    werkzeug \
    gunicorn \
    yt-dlp
RUN pip install --no-cache-dir -U \
    faster-whisper \
    dora-search \
    einops \
    librosa \
    beartype \
    rotary_embedding_torch \
    julius \
    lameenc \
    openunmix \
    qrcode
RUN pip install --no-deps torchaudio
RUN rm -f /opt/conda/lib/python3.11/site-packages/distutils-precedence.pth
RUN pip install -U setuptools
RUN pip install ctranslate2 -U
RUN pip install --no-deps git+https://github.com/facebookresearch/demucs#egg=demucs
RUN git clone https://github.com/shaharpickman555/LyricsProj.git .
COPY ffmpeg .
EXPOSE 8000
CMD ["sh", "-c", "export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH; git pull origin main; pip install -U yt-dlp; exec python frontend.py"]