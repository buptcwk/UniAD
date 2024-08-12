# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},compute,display


SHELL [ "/bin/bash", "--login", "-c" ]

# To fix GPG key error when running apt-get update
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# #install libs first
# RUN apt-get update -q && \
#     apt-get install -q -y \
#     wget \
#     python3.8-dev \
#     python3-pip \
#     python3.8-tk \
#     git \
#     ninja-build \
#     ffmpeg libsm6 libxext6 libglib2.0-0 libsm6 libxrender-dev libxext6 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# intall pytorch
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0+PTX"
# ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV PATH=${PATH}:/usr/local/cuda:/usr/local/cuda/bin

COPY . /root
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update -q && \
    apt-get install -q -y \
    unzip \
    wget \
    && cd /root \
    && wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh

RUN cd /root \
&& bash Anaconda3-2023.07-1-Linux-x86_64.sh -b -p /opt/conda \
&& ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
&& echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
&& source /root/.bashrc \
&& conda create -n uniad python=3.8 -y

SHELL ["conda", "run", "-n", "uniad", "/bin/bash", "-c"]

RUN apt-get update -q && \
    apt-get install -q -y \
    git \
    wget \
    vim \
    libgl1 \
    libglib2.0 \
    build-essential

RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

#ARG SSH_KEY
RUN --mount=type=ssh mkdir -p -m 0700 /root/.ssh \
#RUN echo "$SSH_KEY" > /root/.ssh/id_rsa
#RUN chmod 600 /root/.ssh/id_rsa
#RUN touch /root/.ssh/known_hosts
    && ssh-keyscan gitlab-master.nvidia.com >> /root/.ssh/known_hosts

WORKDIR /root
RUN --mount=type=ssh git clone --branch v1.7.0_h20 --single-branch ssh://git@gitlab-master.nvidia.com:12051/charliech/mmcv.git
RUN cd mmcv && \
    MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=8 python setup.py build_ext && \
    MMCV_WITH_OPS=1 FORCE_CUDA=1 python setup.py develop

RUN pip install mmdet==2.25.1
RUN pip install mmsegmentation==0.29.1

WORKDIR /root
RUN --mount=type=ssh git clone --branch v1.0.0rc4_h20 --single-branch ssh://git@gitlab-master.nvidia.com:12051/charliech/mmdetection3d.git
RUN cd mmdetection3d && \
    pip install -v -e .

WORKDIR /root
RUN --mount=type=ssh git clone --branch main --single-branch ssh://git@gitlab-master.nvidia.com:12051/charliech/uniad.git
RUN cd uniad && \
    pip install -r requirements.txt

RUN cd uniad && \
    mkdir ckpts && \
    cd ckpts && \
    wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth && \
    wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/uniad_base_track_map.pth && \
    wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth  && \
    cd .. && \
    mkdir data && cd data && \
    wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz && \
    wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip && \
    mkdir nuscenes && \
    mv v1.0-mini.tgz nuscenes/ && \
    mv can_bus.zip nuscenes/ && \
    cd nuscenes && \
    tar zxvf v1.0-mini.tgz && \
    unzip -qq -o can_bus.zip && \
    cd maps && \
    wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip && \
    unzip nuScenes-map-expansion-v1.3.zip && \
    cd ../../ && \
    mkdir infos && \
    mkdir others && cd others && \
    wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl

# modify nuscenes/eval/detection/data_classes.py, ref link https://blog.csdn.net/XCCCCZ/article/details/134295931?spm=1001.2014.3001.5501
RUN sed -i -e "39c\\ \ \ \ \ \ \ \ self.class_names = list(self.class_range.keys())" /opt/conda/envs/uniad/lib/python3.8/site-packages/nuscenes/eval/detection/data_classes.py


WORKDIR /root

RUN pip install IPython==8.12.3
RUN pip install nvtx
RUN pip install yapf==0.40.1

# local-rank
