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
## Build and Run Docker
env PLATFORM=`uname -m` TRAVIS_COMMIT=latest ./build.sh ; # 15 min
docker run -it --name deepC-0.13-pypi quay.io/pypa/manylinux2010_x86_64
mkdir /pypi-deepC && cd /pypi-deepC

## Install required packages
export PATH=/opt/python/cp36-cp36m/bin:${PATH}
yum install wget 
pip3 install numpy wheel twine

## Install protobuf (needed by onnx) - 10 mins
yum remove protoc protobuf
git clone https://github.com/google/protobuf.git
cd protobuf
git submodule update --init --recursive
./autogen.sh
./configure --prefix=/usr
make -j8
make check -j8
make install

## Install cmake (needed by onnx) - 5 min
yum remove cmake -y
wget https://cmake.org/files/v3.6/cmake-3.6.2.tar.gz
tar -zxvf cmake-3.6.2.tar.gz
cd cmake-3.6.2
./bootstrap --prefix=/usr/local
make && make install
cd ..

# Install onnx (needed by deepC)
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
python setup.py install


# Install swig (needed by deepC)
wget https://downloads.sourceforge.net/swig/swig-3.0.12.tar.gz
tar xvfz swig-3.0.12.tar.gz
cd swig-3.0.12
./configure --prefix=/usr --without-clisp --without-maximum-compile-warnings --without-pcre && make && make install
cd ..

# Prepare deepC
git clone https://github.com/ai-techsystems/deepC.git
cd deepC
# open setup.py and update 'VERSION=0.XX'
python setup.py bdist_wheel
auditwheel repair dist/deepC*whl

# Test deepC
cd /tmp
python -m pip install /pypi-deepC/deepC/dist/deepC-0.13-cp36-cp36m-linux_x86_64.whl --root .
export PYTHONPATH=/tmp/opt/python/cp36-cp36m/lib/python3.6/site-packages
python -c 'import deepC.dnnc as dc; print(dc.arange(5))'
>>> [0.000000 1.000000 2.000000 3.000000 4.000000]

# Upload deepC
cd /pypi-deepC/deepC/
python3 -m twine upload wheelhouse/deepC*whl
```
