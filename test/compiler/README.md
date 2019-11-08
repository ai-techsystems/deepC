# deepC compiler tests

## Use Model

```onnx2exe model.onnx```

1. *bundleDir :* dirname("generated exe, i.e. a.out or model.exe");
1. *parameter file(s) :* in bundleDir
1. input     file(s) :* with a path relative to current dir.
1. *output    file(s) :* in current dir

## Example

```onnx2cpp mnist.onnx```

```console
> reading onnx model from file  mnist.onnx
> Model info:
>  ir_vesion :  4 
>   doc       : 
> INFO (ONNX): writing model parameter fc.bias to dir ..
> INFO (ONNX): writing model parameter fc.weight to dir ..
> INFO (ONNX): writing model parameter fc2.bias to dir ..
> INFO (ONNX): writing model parameter fc2.weight to dir ..
> running DNNC graph sanity check.
> Writing intermediate file  ./mnist.cpp
> Compiling ...
> INFO (ONNX): model files are ready in dir .
```
