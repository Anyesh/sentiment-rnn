# # Base Image
# FROM python:3.6

# # create and set working directory
# RUN mkdir /app
# WORKDIR /app


# # Add current directory code to working directory
# ADD . /app/

# # set default environment variables
# ENV PYTHONUNBUFFERED 1
# ENV LANG C.UTF-8
# ENV DEBIAN_FRONTEND=noninteractive

# # set project environment variables
# # grab these via Python's os.environ
# # these are 100% optional here
# ENV PORT=8000

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     tzdata \
#     python3-setuptools \
#     python3-pip \
#     python3-dev \
#     python3-venv \
#     git \
#     && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*


# # install environment dependencies
# RUN pip3 install --upgrade pip
# # RUN pip3 install pipenv

# # Install project dependencies
# # RUN pipenv install --skip-lock --system --dev
# RUN pip install -r requirements.txt

# EXPOSE 8080
# # CMD gunicorn wsgi:app --bind 0.0.0.0:$PORT


FROM ubuntu:18.04

# FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y htop python3-dev wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n ml python=3.7

COPY . src/


RUN /bin/bash -c "cd src \
    && source activate ml \
    && pip install -r requirements.txt"

EXPOSE 8080

ENTRYPOINT /bin/bash -c "cd src && source activate ml && python main.py"