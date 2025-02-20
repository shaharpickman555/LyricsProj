FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
USER root
WORKDIR /app

ENV TORCH_HOME=/data/models
ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    openssh-server \
    vim \
    git \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /opt/conda/bin/ffmpeg

RUN mkdir /var/run/sshd
COPY id_rsa.pub /tmp/id_rsa.pub
RUN chmod 755 /root && mkdir /root/.ssh/ && chmod 700 /root/.ssh
RUN cat /tmp/id_rsa.pub >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys

RUN pip install --no-cache-dir -U \
    pip \
    setuptools
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git
RUN pip install --no-cache-dir -U \
    socketio \
    flask-socketio \
    gunicorn \
    yt-dlp \
    faster-whisper \
    dora-search \
    einops \
    julius \
    lameenc \
    openunmix
RUN pip install --no-deps torchaudio
RUN rm -f /opt/conda/lib/python3.11/site-packages/distutils-precedence.pth
RUN pip install -U setuptools
RUN pip install ctranslate2 -U
RUN pip install --no-deps git+https://github.com/facebookresearch/demucs#egg=demucs
RUN git clone https://github.com/shaharpickman555/LyricsProj.git .
EXPOSE 8000
EXPOSE 22
CMD ["sh", "-c", "export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH; service ssh start; git pull origin main; pip install -U yt-dlp; exec python frontend.py"]