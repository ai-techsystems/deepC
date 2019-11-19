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
pip install https://github.com/pypa/manylinux
cd manylinux
PLATFORM=$(uname -m) TRAVIS_COMMIT=latest ./build.sh

yum install wget
wget https://downloads.sourceforge.net/swig/swig-3.0.12.tar.gz
tar xvfz swig-3.0.12.tar.gz
cd swig-3.0.12
./configure --prefix=/usr --without-clisp --without-maximum-compile-warnings --without-pcre && make && make install
pip3 install numpy onnx==1.5.0 wheel twine

python3 setup.py bdist_wheel
auditwheel repair dist/deepC*whl
python3 -m twine upload wheelhouse/deepC*whl
```
