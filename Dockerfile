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
    faster-whisper \
    demucs \
    ffmpeg-python
RUN git clone --depth 1 --branch v4.0.1 --single-branch https://github.com/facebookresearch/demucs /lib/demucs
COPY cmdb.py .
RUN mkdir ./uploads
EXPOSE 8502
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
CMD ["streamlit", "run", "cmdb.py", "--server.port=8502", "--server.headless=true"]
