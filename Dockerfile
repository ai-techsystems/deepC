# base image is ubuntu 18.04
FROM ubuntu:18.04

# update the package-list and install dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  python3.6-dev \
  python3-pip \
  swig \
  doxygen \
  clang-format \
  clang \
  clang-8 \
  llvm-8 \
  llvm-8-dev \
  protobuf-compiler \
  libprotoc-dev \
  vim

# install numpy onnx with pip
# RUN pip3 install numpy onnx
RUN pip3 install numpy==1.15.0 onnx==1.5.0

# copy everything from current directory to container directory
COPY . /dnnCompiler

# below code is commented as we are using bash script to run this
# process because we need to run this differently from root directory and
# from swig directory, but docker doesn't support adding files from
# up a directory. For more information see this link:
# https://stackoverflow.com/questions/27068596/how-to-include-files-outside-of-dockers-build-context

# WORKDIR /dnnCompiler
# CMD make clean \
#   && make