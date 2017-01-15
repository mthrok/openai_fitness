FROM continuumio/anaconda

RUN apt-get update && \
    apt-get -y install \
    g++ \
    git \
    cmake \
    libsdl1.2-dev \
    libsdl-gfx1.2-dev \
    libsdl-image1.2-dev \
    libhdf5-serial-dev \
    xvfb

WORKDIR /src
ADD ./ ./
RUN ./cci_script/install_ale.sh && \
    ./cci_script/download_atari_roms.sh luchador/env/ale/rom && \
    conda install libgcc numpy scipy mkl coverage && \
    pip install git+git://github.com/Theano/Theano.git && \
    pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl && \
    pip install .
CMD ["luchador"]
