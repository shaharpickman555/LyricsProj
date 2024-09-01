FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
USER root
WORKDIR /app

ENV TORCH_HOME=/data/models
ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

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

RUN mkdir /var/run/sshd
COPY id_rsa.pub /tmp/id_rsa.pub
RUN mkdir /root/.ssh/
RUN chmod 700 /root/.ssh
RUN cat /tmp/id_rsa.pub >> /root/.ssh/authorized_keys

RUN pip install --no-cache-dir \
    streamlit \
    yt-dlp \
    faster-whisper
RUN pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs
COPY cmdb.py .
RUN mkdir ./uploads
EXPOSE 8000
EXPOSE 22
CMD ["service", "ssh", "restart"]
CMD ["streamlit", "run", "cmdb.py", "--server.port=8000", "--server.headless=true"]
