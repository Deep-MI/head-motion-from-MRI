#FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
#FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub  
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub 


RUN apt-get update 
RUN apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libsm6 libxext6 libxrender-dev \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
pip3 install scikit-image scikit-learn tensorboard pandas ipdb medcam h5py captum dill typing_extensions 

RUN pip install -U numpy
RUN pip install -U monai
RUN pip install -U opencv-python



#ENTRYPOINT ['/bin/sh']
