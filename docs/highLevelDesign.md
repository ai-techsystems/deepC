# High level architecture

## Front End
This part of the design produces [LLVM 8.0 IR](https://releases.llvm.org/8.0.0/docs/LangRef.html) (Internal Representation) 
without regard to accelerator specific optimization,  which are handled in the back-end support for each device individually.

## ONNX support

While, ONNX has two official ONNX variants; 
1. The neural-network-only **ONNX** and 
2. it's classical Machine Learning extension, **ONNX-ML**.

**DNNC** supports neural-network-only ONNX, which only supports tensors as input and output types *(without support for sequences and maps)*

### Rererence
1. [ONNX support](https://github.com/onnx/onnx/blob/master/docs/IR.md)
2. [LLVM docs](https://releases.llvm.org/8.0.0/docs)
