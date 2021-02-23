FROM continuumio/miniconda3

# Conda environment
COPY . /root/ademfa
WORKDIR /root/ademfa
RUN conda env create -f environment.yml python=3.8.5

# OpenGL
RUN apt-get update && apt-get install libgl1-mesa-glx -yq

# Download model weights
RUN wget --no-check-certificate "https://cloud.isir.upmc.fr/owncloud/index.php/s/M00y5MDWOOwtWBO/download" && tar -xvJf ./download && rm ./download

# Face analysis
ENTRYPOINT ["conda", "run", "-n", "ademfa", "python", "face_analysis.py"]