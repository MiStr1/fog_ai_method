ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION} as base
#FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as base

RUN apt-get update && apt-get install -y curl

ARG USE_PYTHON_3_NOT_2=1
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

# Core linux dependencies.
RUN apt-get install -y \
        vim

# Install decenter package
ENV DECENTER_URL="./decenter-0.5-py3-none-any.whl"
ADD ${DECENTER_URL} /
RUN python -m pip install ${DECENTER_URL}
