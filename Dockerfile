# base Ubuntu Image
FROM ubuntu:20.04

# Python settings to see logs in terminal
ENV PYTHONUNBUFFERED=TRUE

# bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# seting a time zone that prevents docker from hanging at image build
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git \
    libgomp1 \
    && apt-get clean

# Miniforge -> ARM-compatible Miniconda alternative
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# set conda path
ENV PATH /opt/conda/bin:$PATH

# copy files to workdir
WORKDIR /app
COPY . /app

# create env
COPY environment.yml .
RUN conda env create -f environment.yml

# use new env
RUN echo "conda activate segformer_env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# set path to new env
ENV PATH /opt/conda/envs/segformer_env/bin:$PATH
ENV CONDA_DEFAULT_ENV segformer_env

# set the environment variable to fix TLS issue
ENV LD_PRELOAD /usr/lib/aarch64-linux-gnu/libgomp.so.1

# copy the source code
COPY src/app.py entrypoint.sh ./

# make the entrypoint script executable
RUN chmod +x entrypoint.sh


ENTRYPOINT ["./entrypoint.sh"]
