# Generic Steps to build Python Dist Wheel

## Compile and Test

```
% cd <dnnCompiler-repo>
% make clean
% make
% make TEST; # make sure tests are clean.
```

## Build

```
% rm -fr dist
% python setup.py bdist_wheel
```

## pip install locally

```
% cd /tmp
% python -m pip install <dnnCompiler-repo>/dist/deepC-0.1-py3-none-any.whl --root .
```
This will install 
1. python package in /tmp/usr/local/lib/python3.X/dist-packages/deepC
1. binaries in /tmp/usr/local/bin/

## Test the installation.
```
% setenv PYTHONPATH /tmp/usr/local/lib/python3.6/dist-packages
% python -c 'import deepC.dnnc as dc; print(dc.arange(5))'
> [0.000000 1.000000 2.000000 3.000000 4.000000]
```

## Upload on pip
```
% cd <dnnCompiler-repo>
% python -m twine upload dist/*
```


Steps to build Python Dist Wheel on CentOS with manylinux
===================================================
```
docker pull quay.io/pypa/manylinux2014_x86_64
sudo docker run -it quay.io/pypa/manylinux2014_x86_64 /bin/bash

cd /

yum group install "Development Tools"
yum remove swig
yum install python36 python36-devel swig3
pip3 install numpy onnx==1.5.0 wheel twine


git clone https://github.com/ai-techsystems/dnnCompiler.git
cd dnnCompiler
make CC=g++
python3 setup.py bdist_wheel
```
