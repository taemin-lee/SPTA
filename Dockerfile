FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN python3 -m pip install --upgrade pip
RUN apt update \
    && apt install -y git wget unzip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install sacred visdom
RUN pip install git+https://github.com/luizgh/visdom_logger
RUN pip install git+https://github.com/ildoonet/pytorch-randaugment
RUN pip install sklearn

VOLUME ["/workspace"]
WORKDIR /workspace
