FROM jupyter/base-notebook:latest

USER root
RUN apt-get update \
 && apt-get install -y build-essential g++ libgl1-mesa-glx libx11-6 git

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt