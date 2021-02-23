FROM continuumio/miniconda3

# OpenGL
RUN apt-get update && apt-get install libgl1-mesa-glx -yq

# Conda environment
COPY . /root/ademfa
WORKDIR /root/ademfa
SHELL ["/bin/bash", "-c"]
RUN conda env create -f /root/ademfa/environment.yml python=3.8.5
RUN echo "source activate ademfa" > /root/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# Download model weights
RUN wget --no-check-certificate "https://cloud.isir.upmc.fr/owncloud/index.php/s/M00y5MDWOOwtWBO/download" && tar -xvJf ./download && rm ./download
