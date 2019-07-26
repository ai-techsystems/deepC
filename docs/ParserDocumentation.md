# Parser: x to symbol table


## Overview

The first module of the DNN Compiler encompasses conversion of the neural network graph in various formats (.caffe, .pb, .onnx, etc.) to a universal symbol table representation. This symbol table will be used to construct the LLVM IR graph, optimization, and compilation into various backend platforms.

In this section, we discuss parsing of neural network graphs into the symbol table. Thus far, we have added functionality for an ONNX parser, enabling conversion from an ONNX graph (.onnx) to our symbol table (.sym).


## Symbol Table Syntax

This section covers the syntax for our symbol table representation.


### Nodes


```
op_type 
    <name> 
  [ [<input-1> <input-1_attr>]   [<input-2> <input-2_attr>]   ... [<input-n> <input-n_attr>] ]
  [ [<output-1> <output-1_attr>] [<output-1> <output-1_attr>] ... [<output-n> <output-n_attr>] ]
  [ parameters ]
```



### Input/Output (ValueInfoProto)


```
<Input/Output>
  <name>
  [ <dimensions> ]
```



### Initalizers


```
<Initializer>
  <name>
  [ <dims> ]
  [ <data> ]
```



## Sample Symbol Table

This section provides a sample symbol table for a neural network involving the convolution operation.


```
Conv
	""
	[ ["0"] ["weight"] ["bias"] ]
	[ ["3"] ]
	Attr
		"/dilations"
		[ [1, 1] ]
	Attr
		"/group"
		[ 1 ]
	Attr
		"/kernel_shape"
		[ [3, 3] ]
	Attr
		"/pads"
		[ [0, 0, 0, 0] ]
	Attr
		"/strides"
		[ [1, 1] ]

Input
	"0"
	[ 1, 4, 5, 6 ]

Input
	"weight"
	[ 5, 4, 3, 3 ]

Input
	"bias"
	[ 5 ]

Output
	"3"
	[ 1, 5, 3, 4 ]

Initializer
	"bias"
	[ 5 ]
	[ b"\x1e'\x9f=\xd0\xf67<'C\x04\xbe\xb2\x9e\xc7=gG&>" ]

Initializer
	"weight"
	[ 5, 4, 3, 3 ]
	[ b'c9\x9f\xbd...\x86\x81\x94\xbd' ]
```



## ONNX Parser


### ONNX Operators

ONNX operators for which testcases have been generated: \
`add, sub, mul, div, cat, mm, addmm, neg, sqrt, tanh, sigmoid, mean, sum, prod, t, expand, transpose, view, split, squeeze, prelu, threshold, leaky_relu, glu, softmax, avg_pool2d, log_softmax, unfold, elu, concat, abs, index_select, pow, clamp, max, min, eq, gt, lt, ge, le, exp, sin, cos, tan, asin, acos, atan, permute, Conv, BatchNorm, MaxPool1d, MaxPool2d, MaxPool3d, Embedding, RNN, ConstantPad1d, ConstantPad2d, ConstantPad3d`

Testing incomplete:


```
FeatureDropout, Index, Expand, Unfold
```



## Parser Testing


### Testing Phase 1: Unit ONNX operator testcases



*   For each operator in the list below (common PyTorch operators supported in ONNX): created a .onnx binary file and .txt file representing the onnx in human-readable format.

In order to test the ONNX parser and to ensure compatibility of the output symbol table with that of other parsers, we created sample .onnx files with graphs involving common unit operations. For example, we created a file `sigmoid.onnx` for the unit operation of element-wise sigmoid. To assist in the creation of the .onnx file, we first created the graph in PyTorch, then exported the graph to ONNX using the `torch.onnx.export`. However, one drawback of this approach is that only 62 of the 139 ONNX operators in the most recent version. [Link](https://pytorch.org/docs/stable/onnx.html#supported-operators.) for supported ONNX Operators in PyTorch.

Because there was significant redundancy in the creation of PyTorch graphs for various operators, we automated the process such that only a few characteristics of the operator need to be specified, such as operator name, output operation, etc. The output of the automated function includes the python file to create the binary .onnx and readable .txt file for the given operator. [Link](https://github.com/ai-techsystems/dnnCompiler/blob/master/test/pytorch-to-onnx/onnx_generator.py) for automated python file generator.

*   Wrote a generator to create outputs for each individual operator; generator requires only a short string specifying details for the particular test case, and creates a python file that produces .onnx and .txt file.

Example py file, for the max operator:


```
import torch.onnx
import torch.nn as nn
import numpy as np
import onnx

onnx_filename = "./max/max.onnx"
text_filename = "./max/max.txt"

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
	def forward(self, x):
		output = torch.max(x)
		return output

model = Net()
test_input = torch.full([3, 3], 4)
torch.onnx.export(model, test_input, onnx_filename)
with open(text_filename, 'w') as f:
	model = onnx.load(onnx_filename)
	f.write(str(model.graph))

operators = {}
    operators['acos'] = {'output_string':'torch.acos(x)', 'test_input_string':'torch.randn(1, 5)'}
// ...
    operators['Dropout'] = {'output_string': 'torch.nn.Dropout(p=0.3)', 'test_input_string': 'torch.full([4, 4], 3)', 'is_module': True}

//generate .py file
//run .py file → .onnx file, .txt file
```



### Testing Phase 2: Converting test cases to .sym.gold files

Using a https://github.com/ai-techsystems/dnnCompiler/blob/master/test/gold_generator.py, we converted the testcases for unit operators to gold standard files (.sym.gold) for future testing.
