#!/bin/bash
sudo docker build -t dnnc .
sudo docker run dnnc /bin/bash -c \
  "cd /dnnCompiler && make clean && make"
