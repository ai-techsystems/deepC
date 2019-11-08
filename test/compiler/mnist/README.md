
Demo
=======
Checkout [mnist.ir](mnist.ir)

## Step 1: 
Generate intermediate code

```% onnx2cpp mnist.onnx```

## Step 2: 
Optimize and compile

```% /usr/bin/clang++-8 -O3 mnist.cpp -I ../../../include/ -isystem ../../../packages/eigen-eigen-323c052e1731/ -o mnist.exe```

## Step 3: 
Test run

```% ./mnist.exe```

## Step 4: 
Run it in the loop

```% python demo.mnist.py```


More Info
=======

1. batch size 1
1. pytorch version 1.2
1. ONNX IR version 1.4
1. [Training colab notebook](https://colab.research.google.com/drive/1JTcR5A0dQ8y_TKy_DbqpMv1caWjaO_jL)
