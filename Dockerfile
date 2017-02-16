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
WORKDIR /src_
RUN git clone https://github.com/mgbellemare/Arcade-Learning-Environment && \
    cd Arcade-Learning-Environment && \
    cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF . && \
    make && \
    pip install --upgrade .
RUN conda install libgcc numpy scipy mkl coverage flake8 && \
    pip install codacy-coverage Sphinx sphinx_rtd_theme && \
    pip install git+git://github.com/Theano/Theano.git && \
    pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp27-none-linux_x86_64.whl
CMD ["bash"]
