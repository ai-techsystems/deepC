# dnn Compiler
---

## 📛 Introduction 

dnn Compiler is designed to **enable and perform** deep learning neural networks by focussing on features of custom ai-accelerators like FPGAs, eFPGAs and other embedded devices like [raspberry-pi](https://www.raspberrypi.org/), [odroid](https://www.hardkernel.com/), [arduino](https://www.arduino.cc/), [SparkFun Edge](https://www.sparkfun.com/products/15170), [risc-V](https://www.amazon.com/Seeed-Studio-Sipeed-Maixduino-RISC-V/dp/B07SW9ZWQQ) and others.

dnn Compiler is ahead of time compiler producing optimized executable based on [LLVM compiler tool chain](https://llvm.org/) and [openAcc](https://www.openacc.org/) specialized for deep neural networks with [ONNX](https://onnx.ai/) as front end.


## 📝 Design

Main component of **dnn Compiler** has been designed to represent and optimize the common deep learning networks in high level graph IR and to transform the computation graph to minimize memory utilization, optimize data layout and fuse computation patterns for different hardware backends.

<img width="800" alt="Architecture" src="misc/dnnCompilerArch.jpg">

Read more at [high level design document](docs/highLevelDesign.md) 

## 💧 PreRequisites

* [ONNX 1.5](https://github.com/onnx/onnx/tree/rel-1.5.0#installation)
* [LLVM 8.0](http://releases.llvm.org/8.0.0/docs/GettingStarted.html#getting-started-quickly-a-summary)
* [Python 3.6](https://www.python.org/downloads/release/python-360/)
* [SWIG 3.0](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/)

## ⚙ Installation
build and install dnn Compiler locally from source code with following steps

### ⭕ Ubuntu 18.04
You can install ubuntu18.04 on windows [Watch HowTo video here](https://www.youtube.com/watch?v=QbmRXJJKsvs) or [Google it](https://www.google.com/search?q=how+to+setup+ubuntu+on+virtualbox&oq=how+to+setup+ubuntu+on+virtual+box)

Follow the steps to install pre-requisites
```
sudo apt-get update
sudo apt-get install build-essential python3.6-dev python3-pip swig doxygen clang-format clang clang-8 llvm-8 llvm-8-dev
sudo pip3 install numpy
# TODO: libprotobuf-dev protobuf-compiler cmake graphviz libpng-dev wget opencl-headers libgoogle-glog-dev
```

Once you are done, build dnnCompiler
```
git clone https://github.com/ai-techsystems/dnnCompiler.git 
cd dnnCompiler
make
```

#### 📜 Output
```
find include src swig -name \*.h -print0 -o -name \*.cpp -print0 | xargs -0 -P8 -n1 clang-format -i
make -C src
make[1]: Entering directory 'dnnCompiler/src'
make -C core
make[2]: Entering directory 'dnnCompiler/src/core'
compiling broadcast.cpp
/usr/bin/g++ -O3 -Wall -std=c++14 -fPIC -march=native -msse2 \
    -isystem ./packages/eigen-eigen-323c052e1731 -I./include \
    -c broadcast.cpp -o obj/broadcast.o
compiling tensor.cpp
...
...
/usr/bin/g++ -shared  ./obj/dnnc_swig.o ./obj/dnnc_pyutils.o ./obj/dnnc_api.o -o lib/libdnnc.so
ln -s -f lib/libdnnc.so _dnnc.so
/usr/bin/python3 ../test/swig/basic.py
```

## 🏃‍♂️ Usage
```
[build]$ wget https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz
[build]$ tar xvf mnist.tar.gx
[build]$ ./dnnc
```
*Usage:  ./dnnc onnx_model_file*
```
[build]$ ./dnnc mnist/model.onnx
reading CNTK graph from mnist/model.onnx 
populating graph ...
```

---

## ➕ Contribute

dnn Compiler adopts apache committer model, we aim to create an open source project that is maintained and owned by the community. Checkout the Contributor Guide.

## 🙏 Acknowledgement 
We acknowledge the efforts predecessor projects like [LLVM](https://llvm.org/), [ONNX](https://onnx.ai/) etc. to make this project a reality.

---

## 🕵️‍♂️ Why compiler❔
dnnCompiler is targeted towards devices with small formfactor like microcontrollers, which are part of all sorts of household devices: think appliances, cars, and toys. In fact, there are around 30 billion microcontroller-powered devices produced each year. They're cheap, require very little energy, and are very reliable. 

By bringing deep learning models to tiny microcontrollers, we can boost the intelligence of billions of devices that we use in our lives, without relying on expensive hardware or reliable internet connections. Imagine smart appliances that can adapt to your daily routine, intelligent industrial sensors that understand the difference between problems and normal operation, and magical toys that can help kids learn in fun and delightful ways.


**🚧 Project Under Development.** *Stay tuned. We plan to release the first version in the third quarter of 2019.*
