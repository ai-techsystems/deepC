#!/bin/bash
cd ..
sudo docker build -t dnnc .
sudo docker run dnnc /bin/bash -c \
  "cd /dnnCompiler/swig && make clean && make"