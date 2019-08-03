
# Copyright 2018 The AITS DNNC Authors. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
#
# This file is part of DNN compiler maintained at 
# https://github.com/ai-techsystems/dnnCompiler

from generate_output import *

operators = {}
operators['Abs'] = {
	'nodes':['helper.make_node("Abs", ["0"], ["1"])'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Acos'] = {
	'nodes':['helper.make_node("Acos", ["0"], ["1"])'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Acosh'] = {
	'nodes':['helper.make_node("Acosh", ["0"], ["1"])'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Add'] = {
	'nodes':['helper.make_node("Add", ["A","B"], ["C"])'],
	'inputs':'[helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 4)),helper.make_tensor_value_info("B", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("C", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['And'] = {
	'nodes':['helper.make_node("And", ["A","B"], ["C"])'],
	'inputs':'[helper.make_tensor_value_info("A", TensorProto.BOOL, (2, 3, 4)),helper.make_tensor_value_info("B", TensorProto.BOOL, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("C", TensorProto.BOOL, (2, 3, 4))]'
}

operators['ArgMax'] = {
	'nodes': ['helper.make_node("ArgMax", ["0"], ["1"], axis = 0 , keepdims = 1)'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['ArgMin'] = {
	'nodes': ['helper.make_node("ArgMin", ["0"], ["1"], axis = 0 , keepdims = 1)'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Asin'] = {
	'nodes':['helper.make_node("Asin", ["0"], ["1"])'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Asinh'] = {
	'nodes':['helper.make_node("Asinh", ["0"], ["1"])'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Atan'] = {
	'nodes':['helper.make_node("Atan", ["0"], ["1"])'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Atanh'] = {
	'nodes':['helper.make_node("Atanh", ["0"], ["1"])'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['AveragePool'] = {
'nodes': ['helper.make_node("AveragePool", ["0"], ["1"], kernel_shape=[2,2])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 3, 32, 32))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (1, 3, 31, 31))]'
}

operators['BatchNormalization'] = {
'nodes': ["helper.make_node('BatchNormalization',inputs=['x', 's', 'bias', 'mean', 'var'],outputs=['y'])"],
'inputs':'[helper.make_tensor_value_info("x", TensorProto.FLOAT, (2, 3)), helper.make_tensor_value_info("s", TensorProto.FLOAT, (1,2)), helper.make_tensor_value_info("bias", TensorProto.FLOAT, (1,2)), helper.make_tensor_value_info("mean", TensorProto.FLOAT, (1,2)), helper.make_tensor_value_info("var", TensorProto.FLOAT, (1,2))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3))]',
}

operators['BitShift'] = {
	'nodes':['helper.make_node("BitShift", ["A","B"], ["C"], direction="LEFT")'],
	'inputs':'[helper.make_tensor_value_info("A", TensorProto.UINT16, (2, 3, 4)),helper.make_tensor_value_info("B", TensorProto.UINT16, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("C", TensorProto.UINT16, (2, 3, 4))]'
}

operators['Cast'] = {
	'nodes':['helper.make_node("Cast", ["input"], ["output"], to = 1)'],
	'inputs':'[helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Ceil'] = {
	'nodes':['helper.make_node("Ceil", ["0"], ["1"])'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Clip'] = {
	'nodes':['helper.make_node("Clip", ["0"], ["1"], min= -1.0, max= 1.0)'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Concat'] = {
	'nodes': ['helper.make_node("Concat", ["0"], ["1"], axis = 0)'],
	'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Constant'] = {
	'nodes':['helper.make_node("Constant", [], ["1"], value = onnx.helper.make_tensor(name="const_tensor", data_type=onnx.TensorProto.FLOAT, dims=values.shape, vals=values.flatten().astype(float)))'],
	'inputs':'[]',
	'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (5,5))]',
	'declarations':['values=np.random.randn(5, 5).astype(np.float32)']
}

operators['Compress'] = {
	'nodes': ['helper.make_node("Compress", ["input", "condition"], ["output"] , axis = 1)'],
	'inputs':'[helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("condition", TensorProto.FLOAT, (2, 3, 4)) ]',
	'outputs':'[helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['ConstantOfShape'] = {
'nodes':['helper.make_node("ConstantOfShape", ["0"], ["1"], value=helper.make_tensor("value", TensorProto.FLOAT, [1], [6]))'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Conv'] = {
'nodes':['helper.make_node("Conv", ["0", "1"], ["2"], dilations=[1, 1], kernel_shape=[3, 3])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 1, 6, 6)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (1, 1, 3, 3))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (1, 1, 4, 4))]'
}

operators['ConvInteger'] = {
'nodes':['helper.make_node("ConvInteger", ["0", "1"], ["2"], dilations=[1, 1], kernel_shape=[3, 3])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 1, 6, 6)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (1, 1, 3, 3))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (1, 1, 4, 4))]'
}

operators['ConvTranspose'] = {
'nodes':['helper.make_node("ConvTranspose", ["0", "1"], ["2"], dilations=[1, 1], kernel_shape=[3, 3])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 1, 3, 3)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (1, 2, 3, 3))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (1, 2, 5, 5))]'
}

operators['Cos'] = {
'nodes':['helper.make_node("Cos", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Cosh'] = {
'nodes':['helper.make_node("Cosh", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['CumSum'] = {
'nodes':['helper.make_node("CumSum", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (5,))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (5,))]'
} # ONNX doesn’t seem to recognize opset 11 operators, even though they are listed on the operator page

operators['DepthToSpace'] = {
'nodes':['helper.make_node("DepthToSpace", ["0"], ["1"], blocksize=2)'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 4, 2, 3))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (1, 1, 4, 6))]'
}

operators['DequantizeLinear'] = {
'nodes':['helper.make_node("DequantizeLinear", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (5,)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (1,))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (5,))]'
}

operators['Div'] = {
'nodes':['helper.make_node("Div", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Dropout'] = {
'nodes':['helper.make_node("Dropout", ["0"], ["1"], ratio=0.3)'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (4,))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (4,))]'
}

operators['Elu'] = {
'nodes':['helper.make_node("Elu", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (4,))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (4,))]'
}

operators['Equal'] = {
'nodes':['helper.make_node("Equal", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Erf'] = {
'nodes':['helper.make_node("Erf", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Exp'] = {
'nodes':['helper.make_node("Exp", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Expand'] = {
'nodes':['helper.make_node("Expand", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (3, 1)), helper.make_tensor_value_info("1", TensorProto.INT64, (2, 1, 6))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 1, 6))]'
}

operators['EyeLike'] = {
'nodes':['helper.make_node("EyeLike", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (5, 5))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (5, 5))]'
}

operators['Flatten'] = {
'nodes':['helper.make_node("Flatten", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (5, 4, 3, 2))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (5, 24))]'
}

operators['Floor'] = {
'nodes':['helper.make_node("Floor", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['GRU'] = {
'nodes':['helper.make_node("GRU", ["0", "1", "2"], ["", "3"], hidden_size=5)'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 3, 2)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (1, 3 * 5, 2)), helper.make_tensor_value_info("2", TensorProto.FLOAT, (1, 3 * 5, 5))]',
'outputs':'[helper.make_tensor_value_info("3", TensorProto.FLOAT, (1, 3, 5))]'
}

operators['Gather'] = {
'nodes':['helper.make_node("Gather",["data","indices"],["y"],axis=0)'],
'inputs':'[helper.make_tensor_value_info("data", TensorProto.FLOAT, (3, 2)), helper.make_tensor_value_info("indices", TensorProto.INT64, (2,2))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(2,2,2))]'
}

operators['Gemm'] = {
'nodes':['helper.make_node("Gemm",["A","B","C"],["Y"],alpha=1.22,beta=5.3,transA=1,transB=0)'],
'inputs':'[helper.make_tensor_value_info("A", TensorProto.FLOAT, (5, 3)), helper.make_tensor_value_info("B", TensorProto.FLOAT, (5,3)),helper.make_tensor_value_info("C", TensorProto.FLOAT, (3,3))]',
'outputs':'[helper.make_tensor_value_info("Y",TensorProto.FLOAT,(3,3))]'
}

operators['GlobalAveragePool'] = {
'nodes':['helper.make_node("GlobalAveragePool",["x"],["y"])'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(2,3,4,5))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(2,3,1,1))]'
}

operators['GlobalLpPool'] = {
'nodes':['helper.make_node("GlobalLpPool",["x"],["y"],p=3)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(2,3,4,5))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(2,3,1,1))]'
}

operators['GlobalMaxPool'] = {
'nodes':['helper.make_node("GlobalMaxPool",["x"],["y"])'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(2,3,4,5))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(2,3,1,1))]'
}

operators['Greater'] = {
'nodes':['helper.make_node("Greater",["A","B"],["C"])'],
'inputs':'[helper.make_tensor_value_info("A",TensorProto.FLOAT,(2,3)), helper.make_tensor_value_info("B",TensorProto.FLOAT,(2,3))]',
'outputs':'[helper.make_tensor_value_info("C",TensorProto.FLOAT,(2,3))]'
}

operators['HardSigmoid'] = {
'nodes':['helper.make_node("HardSigmoid",["X"],["Y"],alpha=0.2,beta=0.2)'],
'inputs':'[helper.make_tensor_value_info("X",TensorProto.FLOAT,(2,3))]',
'outputs':'[helper.make_tensor_value_info("Y",TensorProto.FLOAT,(2,3))]'
}

operators['Hardmax'] = {
'nodes':['helper.make_node("Hardmax",["X"],["Y"],axis=1)'],
'inputs':'[helper.make_tensor_value_info("X",TensorProto.FLOAT,(2,3))]',
'outputs':'[helper.make_tensor_value_info("Y",TensorProto.FLOAT,(2,3))]'
}

operators['Identity'] = {
'nodes':['helper.make_node("Identity",["X"],["Y"])'],
'inputs':'[helper.make_tensor_value_info("X",TensorProto.FLOAT,(3,3,5))]',
'outputs':'[helper.make_tensor_value_info("Y",TensorProto.FLOAT,(3,3,5))]'
}

operators['If'] = {
'nodes':['helper.make_node("If", ["0"], ["1"], else_branch=sub_graph, then_branch=sub_graph)'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.BOOL, [1]), helper.make_tensor_value_info("2", TensorProto.BOOL, [1])]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.BOOL, [1])]',
'declarations':['sub_graph = helper.make_graph([helper.make_node("And", ["0", "2"], ["1"])], "graph", inputs, outputs)']
}

operators['InstanceNormalization'] = {
'nodes':['helper.make_node("InstanceNormalization",["x","s","bias"],["y"],epsilon=1e-05)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(3,4,2,2)), helper.make_tensor_value_info("s",TensorProto.FLOAT,(3,)), helper.make_tensor_value_info("bias",TensorProto.FLOAT,(3,))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(3,4,2,2))]'
}

operators['IsInf'] = {
'nodes':['helper.make_node("IsInf",["X"],["Y"],detect_negative=1,detect_positive=1)'],
'inputs':'[helper.make_tensor_value_info("X",TensorProto.FLOAT,(1,))]',
'outputs':'[helper.make_tensor_value_info("Y",TensorProto.BOOL,(1,))]'
}

operators['IsNaN'] = {
'nodes':['helper.make_node("IsNaN",["X"],["Y"])'],
'inputs':'[helper.make_tensor_value_info("X",TensorProto.FLOAT,(1,))]',
'outputs':'[helper.make_tensor_value_info("Y",TensorProto.BOOL,(1,))]'
}

operators['LRN'] = {
'nodes':['helper.make_node("LRN",["x"],["y"],alpha = 0.0001,beta = 0.75,bias = 1.0,size = 3)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(2,3,4,5))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(2,3,4,5))]'
}

operators['LSTM'] = {
'nodes':['helper.make_node("LSTM",["x","w","r"],["y"],activations=["sigmoid","tanh","relu"],direction="reverse",hidden_size=3)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(3,3,2)),helper.make_tensor_value_info("w",TensorProto.FLOAT,(1,4*3,2)),helper.make_tensor_value_info("r",TensorProto.FLOAT,(1,4*3,3))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(3,1,3,3))]'
}

operators['LeakyRelu'] = {
'nodes':['helper.make_node("LeakyRelu",["x"],["y"],alpha=0.1)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(1,))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(1,))]'
}

operators['Less'] = {
'nodes':['helper.make_node("Less",["x","y"],["z"])'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(1,)),helper.make_tensor_value_info("y",TensorProto.FLOAT,(1,))]',
'outputs':'[helper.make_tensor_value_info("z",TensorProto.BOOL,(1,))]'
}

operators['Log'] = {
'nodes':['helper.make_node("Log",["x"],["y"])'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(1,))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(1,))]'
}

operators['LogSoftmax'] = {
'nodes':['helper.make_node("LogSoftmax",["x"],["y"],axis=1)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(2,3))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(2,3))]'
}

operators['Loop'] = {
'nodes':['helper.make_node("Loop", ["0", "", "1", "3"], ["2"], body=graph)'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.INT64, (1,)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (4, 5)),helper.make_tensor_value_info("3", TensorProto.FLOAT, (4, 5))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (4, 5))]',
'declarations': ['graph = helper.make_graph([helper.make_node("Add", ["1", "3"], ["2"])], "graph", inputs, outputs)']
}

operators['LpNormalization'] = {
'nodes':[' helper.make_node("LpNormalization", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['LpPool'] = {
'nodes':[' helper.make_node("LpPool", ["0"], ["1"], kernel_shape=[2])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4, 1))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['MatMul'] = {
'nodes':['helper.make_node("MatMul", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 4, 3))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 3, 3))]'
}

operators['MatMulInteger'] = {
'nodes':['helper.make_node("MatMulInteger", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.INT64, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.INT64, (2, 4, 3))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.INT64, (2, 3, 3))]'
}

operators['Max'] = {
'nodes':['helper.make_node("Max", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['MaxPool'] = {
'nodes':[' helper.make_node("MaxPool", ["0"], ["1"], kernel_shape=[2])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 3, 32))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (1, 3, 31))]'
}

operators['MaxRoiPool'] = {
'nodes':[' helper.make_node("MaxRoiPool", ["0","1"], ["2"], pooled_shape=[4,5])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4, 5)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (3,5))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (3, 3, 1, 1))]'
}

operators['MaxUnpool'] = {
'nodes':[' helper.make_node("MaxUnpool", ["0","1"], ["2"], kernel_shape=[2])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4, 5)),helper.make_tensor_value_info("1", TensorProto.INT64, (2, 3, 4, 5))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 3, 4, 5))]'
}

operators['Mean'] = {
'nodes':['helper.make_node("Mean", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Min'] = {
'nodes':['helper.make_node("Min", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Mod'] = {
'nodes':[' helper.make_node("Mod", ["0","1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Mul'] = {
'nodes':[' helper.make_node("Mul", ["0","1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Multinomial'] = {
'nodes':[' helper.make_node("Multinomial", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3))]'
}

operators['Neg'] = {
'nodes':[' helper.make_node("Neg", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['NonMaxSuppression'] = {
'nodes':[' helper.make_node("NonMaxSuppression", ["0","1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 3))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (4, 3))]'
}

operators['NonZero'] = {
'nodes':[' helper.make_node("NonZero", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Not'] = {
'nodes':['helper.make_node("Not", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.BOOL, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.BOOL, (2, 3, 4))]'
}

operators['OneHot'] = {
'nodes':['helper.make_node("OneHot", ["0", "1","2"], ["3"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (10,)), helper.make_tensor_value_info("2", TensorProto.FLOAT, (1, 3))]',
'outputs':'[helper.make_tensor_value_info("3", TensorProto.FLOAT, (3, 4, 5))]'
}

operators['Or'] = {
'nodes':['helper.make_node("Or", ["0","1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.BOOL, (2, 3, 4)),helper.make_tensor_value_info("1", TensorProto.BOOL, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.BOOL, (2, 3, 4))]'
}

operators['PRelu'] = {
'nodes':[' helper.make_node("PRelu", ["0","1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 3))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['QLinearConv'] = {
'nodes':['helper.make_node("QLinearConv",["x", "x_scale", "x_zero_point", "w", "w_scale", "w_zero_point", "y_scale", "y_zero_point"],["y"])'],
'inputs':'[helper.make_tensor_value_info("x", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("x_zero_point", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("w", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("w_scale", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("w_zero_point", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("y_zero_point", TensorProto.FLOAT, (2,3,4))]',
'outputs':'[helper.make_tensor_value_info("y", TensorProto.FLOAT, (2,3,4))]'
}

operators['Pow'] = {
'nodes':['helper.make_node("Pow",["X","Y"],["Z"],)'],
'inputs':'[helper.make_tensor_value_info("X",TensorProto.FLOAT,(2,3,4)), helper.make_tensor_value_info("Y", TensorProto.FLOAT,(2,3,4))]',
'outputs':'[helper.make_tensor_value_info("Z",TensorProto.FLOAT,(2,3,4))]'
}

operators['QLinearMatMul'] = {
'nodes':['helper.make_node("QLinearMatMul",["a", "a_scale", "a_zero_point", "b", "b_scale", "b_zero_point", "y_scale", "y_zero_point"],["y"],)'],
'inputs':'[helper.make_tensor_value_info("a", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("a_scale", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("a_zero_point", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("b", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("b_scale", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("b_zero_point", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("y_zero_point", TensorProto.FLOAT, (2,3,4))]',
'outputs':'[helper.make_tensor_value_info("y", TensorProto.FLOAT, (2,3,4))]'
}

operators['QuantizeLinear']= {
'nodes':['helper.make_node("QuantizeLinear",["x","y_scale","y_zero_point"],["y"])'],
'inputs':'''[helper.make_tensor_value_info("x",TensorProto.FLOAT,(2,3,4)),
	helper.make_tensor_value_info("y_scale",TensorProto.FLOAT,(2,3,4)),
	helper.make_tensor_value_info("y_zero_point",TensorProto.FLOAT,(2,3,4))]''',
'outputs':'[helper.make_tensor_value_info("y", TensorProto.FLOAT, (2,3,4))]'
}

operators['RNN'] = {
'nodes':['helper.make_node("RNN",["X","W","R","B"],["","Y"])'],
'inputs':'[helper.make_tensor_value_info("X", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("W", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("R", TensorProto.FLOAT, (2,3,4)),\
	helper.make_tensor_value_info("B", TensorProto.FLOAT, (2,3,4))]',
'outputs':'[helper.make_tensor_value_info("Y",TensorProto.FLOAT,(1,2))]'
}

operators['Reciprocal'] = {
'nodes':['helper.make_node("Reciprocal",["X"],["Y"],)'],
'inputs':'[helper.make_tensor_value_info("X",TensorProto.FLOAT,(1,2))]',
'outputs':'[helper.make_tensor_value_info("Y",TensorProto.FLOAT,(2,3))]'
}

operators['ReduceL1'] = {
'nodes':['helper.make_node("ReduceL1",["D"],["R"],axes=[0],keepdims=1)'],
'inputs':'[helper.make_tensor_value_info("D",TensorProto.FLOAT,(1,2,3))]',
'outputs':'[helper.make_tensor_value_info("R",TensorProto.FLOAT,(1,2,3))]'
}

operators['RandomUniformLike']={
'nodes':['helper.make_node("RandomUniformLike",inputs=["x"],outputs=["y"],low=0.0,high=1.0,dtype=TensorProto.FLOAT)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(1,2,))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(1,2,))]'
}

operators['ReduceL2'] = {
'nodes':['helper.make_node("ReduceL2",["D"],["R"],axes=[0],keepdims=1)'],
'inputs':'[helper.make_tensor_value_info("D",TensorProto.FLOAT,(1,2,3))]',
'outputs':'[helper.make_tensor_value_info("R",TensorProto.FLOAT,(1,2,3))]'
}

operators['ReduceLogSum'] = {
'nodes':['helper.make_node("ReduceLogSum",["D"],["R"],axes=[0,1],keepdims=0)'],
'inputs':'[helper.make_tensor_value_info("D",TensorProto.FLOAT,(1,2,3))]',
'outputs':'[helper.make_tensor_value_info("R",TensorProto.FLOAT,(1,2))]'
}

operators['ReduceLogSumExp'] = {
'nodes':['helper.make_node("ReduceLogSumExp",["D"],["R"],axes=[0],keepdims=1)'],
'inputs':'[helper.make_tensor_value_info("D",TensorProto.FLOAT,(1,2,3))]',
'outputs':'[helper.make_tensor_value_info("R",TensorProto.FLOAT,(1,2,3))]'
}

operators['RandomUniform']={
'nodes':['helper.make_node("RandomUniform",inputs=["x"],outputs=["y"],low=0.0,high=1.0,dtype=TensorProto.FLOAT)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(1,2,))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(1,2,))]'
}

operators['ReduceMax'] = {
'nodes':['helper.make_node("ReduceMax",["D"],["R"],axes=[0],keepdims=1)'],
'inputs':'[helper.make_tensor_value_info("D",TensorProto.FLOAT,(1,2,3))]',
'outputs':'[helper.make_tensor_value_info("R",TensorProto.FLOAT,(1,2,3))]'
}

operators['ReduceMean'] = {
'nodes':['helper.make_node("ReduceMean",["D"],["R"],axes=[0],keepdims=1)'],
'inputs':'[helper.make_tensor_value_info("D",TensorProto.FLOAT,(1,2,3))]',
'outputs':'[helper.make_tensor_value_info("R",TensorProto.FLOAT,(1,2,3))]'
}

operators['Pad'] = {
'nodes':['helper.make_node("Pad",["x"],["y"],mode="constant",pads=[0, 0, 1, 3, 0, 0, 2, 4])'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(1, 3, 4, 5))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(1, 3, 7, 12))]'
}

operators['RandomNormalLike']={
'nodes':['helper.make_node("RandomNormalLike",inputs=["x"],outputs=["y"],mean=0.0,scale=1.0)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(1, 2))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(1, 2))]'
}

operators['ReduceMin'] = {
'nodes':['helper.make_node("ReduceMin",["D"],["R"],axes=[0],keepdims=1)'],
'inputs':'[helper.make_tensor_value_info("D",TensorProto.FLOAT,(1,2,3))]',
'outputs':'[helper.make_tensor_value_info("R",TensorProto.FLOAT,(1,2,3))]'
}

operators['RandomNormal']={
'nodes':['helper.make_node("RandomNormal",inputs=["x"],outputs=["y"],mean=0.0,scale=1.0)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(1,2,))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(1,2,))]'
}

operators['ReduceProd'] = {
'nodes':['helper.make_node("ReduceProd",["D"],["R"],axes=[0],keepdims=1)'],
'inputs':'[helper.make_tensor_value_info("D",TensorProto.FLOAT,(1,2,3))]',
'outputs':'[helper.make_tensor_value_info("R",TensorProto.FLOAT,(1,2,3))]'
}

operators['ReduceSum'] = {
'nodes':['helper.make_node("ReduceSum",["D"],["R"],axes=[0],keepdims=1)'],
'inputs':'[helper.make_tensor_value_info("D",TensorProto.FLOAT,(1,2,3))]',
'outputs':'[helper.make_tensor_value_info("R",TensorProto.FLOAT,(1,2,3))]'
}

operators['ReduceSumSquare'] = {
'nodes':['helper.make_node("ReduceSumSquare", ["0"], ["1"], keepdims=1)'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (3, 2, 2))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (1, 1, 1))]'
}

operators['Relu'] = {
'nodes':['helper.make_node("Relu",["x"],["y"])'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(2,3,4))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(2,3,4))]'
}

operators['Reshape'] = {
'nodes':['helper.make_node("Reshape", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("1", TensorProto.INT64, (2,))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (3, 8))]'
}

operators['Resize'] = {
'nodes':['helper.make_node("Resize", ["0", "1"], ["2"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 1, 2, 4)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (4,))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (1, 1, 1, 2))]'
}

operators['ReverseSequence'] = {
'nodes':['helper.make_node("ReverseSequence", ["0", "1"], ["2"], batch_axis=1, time_axis=0)'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (4, 4)), helper.make_tensor_value_info("1", TensorProto.INT64, (4,))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (4, 4))]'
}

operators['RoiAlign'] = {
'nodes':['helper.make_node("RoiAlign", ["0", "1", "2"], ["3"], output_height=5, output_width=5, sampling_ratio=2)'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 1, 10, 10)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (3, 4)), helper.make_tensor_value_info("2", TensorProto.INT64, (3,))]',
'outputs':'[helper.make_tensor_value_info("3", TensorProto.FLOAT, (3, 1, 5, 5))]'
}

operators['Round'] = {
'nodes': ['helper.make_node("Round", inputs=["0"], outputs=["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (1, 4))]'
}

operators['Scan'] = {
'nodes':['onnx.helper.make_node("Scan",inputs=["initial", "x"],outputs=["y", "z"],num_scan_inputs=1,body=scan_body)'],
'inputs':'[helper.make_tensor_value_info("initial",TensorProto.FLOAT,(1,2)), helper.make_tensor_value_info("x",TensorProto.FLOAT,(1,3,2))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(1,2)), helper.make_tensor_value_info("z",TensorProto.FLOAT,(1,3,2))]',
'declarations':[
'sum_in = onnx.helper.make_tensor_value_info("sum_in", onnx.TensorProto.FLOAT, [2])',
"next = onnx.helper.make_tensor_value_info('next', onnx.TensorProto.FLOAT, [2])",
"sum_out = onnx.helper.make_tensor_value_info('sum_out', onnx.TensorProto.FLOAT, [2])",
"scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [2])",
"add_node = onnx.helper.make_node('Add',inputs=['sum_in', 'next'],outputs=['sum_out'])",
"id_node = onnx.helper.make_node('Identity',inputs=['sum_out'],outputs=['scan_out'])",
"scan_body = onnx.helper.make_graph([add_node, id_node],'scan_body',[sum_in, next],[sum_out, scan_out])"]
}

operators['Scatter'] = {
'nodes':["onnx.helper.make_node('Scatter',inputs=['data', 'indices', 'updates'],outputs=['y'],axis=1)"],
'inputs':'[helper.make_tensor_value_info("data",TensorProto.FLOAT,(1,6)), helper.make_tensor_value_info("indices",TensorProto.INT32,(1,3)), helper.make_tensor_value_info("updates",TensorProto.FLOAT,(1,2))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(1,5))]'
}

operators['Selu'] = {
'nodes':["onnx.helper.make_node('Selu',inputs=['x'],outputs=['y'],alpha=2.0,gamma=3.0)"],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(1,3))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(1,3))]'
}

operators['Shape'] = {
'nodes':["onnx.helper.make_node('Shape',inputs=['x'],outputs=['y'])"],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(3,4,5))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.INT32,(3,4,5))]'
}

operators['Sigmoid'] = {
'nodes':['helper.make_node("Sigmoid", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Sin'] = {
'nodes':['helper.make_node("Sin", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}
operators['Shrink'] = {
'nodes':['helper.make_node("Shrink", ["0"], ["1"], bias = 0.0 , lambd = 0.5)'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}
operators['Size'] = {
'nodes':['helper.make_node("Size", ["0"] , ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Slice'] = {
'nodes':['''helper.make_node("Slice",["data", 
                              "starts",
                              "ends",
                              "axes",
                              "steps"],["output"],
                              )'''],
'inputs':'''[helper.make_tensor_value_info("data", TensorProto.FLOAT, (2, 3, 4)),
       	 helper.make_tensor_value_info("starts", TensorProto.FLOAT, (2, 3, 4)),
         helper.make_tensor_value_info("ends", TensorProto.FLOAT, (2, 3, 4)),
         helper.make_tensor_value_info("axes", TensorProto.FLOAT, (2, 3, 4)),
         helper.make_tensor_value_info("steps", TensorProto.FLOAT, (2, 3, 4))]''',

'outputs':'[helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Sign'] = {
'nodes':['helper.make_node("Sign", ["0"] , ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Sinh'] = {
'nodes':['helper.make_node("Sinh", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Softmax'] = {
'nodes':['helper.make_node("Softmax",["x"],["y"],axis=1)'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(2,3,4))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(2,3,4))]'
}

operators['Softplus'] = {
'nodes':['helper.make_node("Softplus",["x"],["y"])'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(2,3,4))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(2,3,4))]'
}

operators['Sub'] = {
'nodes':['helper.make_node("Sub", ["A","B"], ["C"])'],
'inputs':'[helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 4))], [helper.make_tensor_value_info("B", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("C", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Tan'] = {
'nodes':['helper.make_node("Tan", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (4, 5, 6))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (4, 5, 6))]'
}

operators['Tanh'] = {
'nodes':['helper.make_node("Tanh", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 3))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 3))]'
}

operators['Sqrt'] = {
'nodes':['helper.make_node("Sqrt", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (4, 5, 6))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (4, 5, 6))]'
}

operators['Squeeze'] = {
'nodes':['helper.make_node("Squeeze", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1,3,4,5))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (1,3,4,5))]'
}

operators['Unsqueeze'] = {
'nodes':['helper.make_node("Unsqueeze", ["0"], ["1"], axes=[])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (3,4,5))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (3,4,5))]'
}

operators['Sum'] = {
'nodes':['helper.make_node("Sum", ["0", "1"], ["3"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (4,3,2)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (4,3,2))', 
'outputs':'[helper.make_tensor_value_info("3", TensorProto.FLOAT, (4,3,2))]'
}

operators['Xor'] = {
'nodes':['helper.make_node("Xor", ["x", "y"], ["z"])'],
'inputs':'[helper.make_tensor_value_info("x", TensorProto.BOOL, (2, 3)), helper.make_tensor_value_info("y", TensorProto.BOOL, (2, 3))]',
'outputs':'[helper.make_tensor_value_info("z", TensorProto.BOOL, (2, 3))]'
}

operators['Softsign'] = {
'nodes':['helper.make_node("Softsign",["x"],["y"])'],
'inputs':'[helper.make_tensor_value_info("x",TensorProto.FLOAT,(3,4,5))]',
'outputs':'[helper.make_tensor_value_info("y",TensorProto.FLOAT,(3,4,5))]'
}

operators['ThresholdedRelu'] = {
'nodes':[' helper.make_node("ThresholdedRelu", ["0"], ["1"], alpha=2.0)'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (3,4,5))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (3,4,5))]'
}

operators['Upsample'] = {
'nodes':['helper.make_node("Upsample", ["0","1"], ["2"], mode="nearest")'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2,2)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (1,4))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (4,6))]'
}

operators['TopK'] = {
'nodes':[' helper.make_node("TopK", ["0","1"], ["2","3"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2,3,4)), helper.make_tensor_value_info("1", TensorProto.INT64, (1,1))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2,3,4)), helper.make_tensor_value_info("3", TensorProto.FLOAT, (2,3,4))]'
}

operators['Split'] = {
'nodes':[' helper.make_node("Split", ["0"], ["1","2"]), axis=1 '],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2,6)), helper.make_tensor_value_info("1", TensorProto.FLOAT, (2,3))]',
'outputs':'[helper.make_tensor_value_info("2", TensorProto.FLOAT, (2,3))]'
}

operators['Transpose'] = {
'nodes':['helper.make_node("Transpose", ["A"],["B"])'],
'inputs': '[helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("B", TensorProto.FLOAT, (3, 2, 4))]'
}

operators['Where'] = {
'nodes':['helper.make_node("Where", ["A","B","C"], ["D"])'],
'inputs': '[helper.make_tensor_value_info("A", TensorProto.BOOL, (2, 3, 4)), helper.make_tensor_value_info("B", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("C", TensorProto.FLOAT, (2, 3, 4))]',
'outputs':'[helper.make_tensor_value_info("D", TensorProto.FLOAT, (2, 3, 4))]'
}

operators['Tile'] = {
'nodes':['helper.make_node("Tile", ["A","B"],["C"])'],
'inputs': ' [helper.make_tensor_value_info("A", TensorProto.FLOAT, (2, 3, 4)), helper.make_tensor_value_info("B", TensorProto.INT64, (2, 2))]',
'outputs':'[helper.make_tensor_value_info("C", TensorProto.FLOAT, (2, 3, 4)*2)]'
}

operators['StringNormalizer'] = {
'nodes': ['helper.make_node("StringNormalizer", ["0"], ["1"])'],
'inputs':'[helper.make_tensor_value_info("0", TensorProto.STRING, (1, 4))]',
'outputs':'[helper.make_tensor_value_info("1", TensorProto.STRING, (1, 3))]'
}

operators['TfIdfVectorizer'] = {
'nodes': ['helper.make_node("TfIdfVectorizer", ["0"], ["1"], mode="TF", min_gram_length=2, max_gram_length=2, max_skip_count=0, ngram_counts=ngram_counts, ngram_indexes=ngram_indexes, pool_int64s=pool_int64s)'],
'inputs': '[helper.make_tensor_value_info("0", TensorProto.FLOAT, (2, 6))]',
'outputs': '[helper.make_tensor_value_info("1", TensorProto.FLOAT, (2, 7))]',
'declarations': ['ngram_counts = np.array([0, 4]).astype(np.int64)',
 'ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)',
 'pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(np.int64)']
}

operators['SpaceToDepth'] = {
'nodes':['helper.make_node("SpaceToDepth", ["0"], ["1"], blocksize=2)'],'inputs':'[helper.make_tensor_value_info("0", TensorProto.FLOAT, (1, 1, 4, 6))]','outputs':'[helper.make_tensor_value_info("1", TensorProto.FLOAT, (1, 4, 2, 3))]'
}

for operator in operators.keys():
	operator_info = operators[operator]
	declarations = []
	if 'declarations' in operators[operator].keys():
		declarations = operators[operator]['declarations']
	node_params = []
	if 'node_params' in operators[operator].keys():
		node_params = operators[operator]['node_params']
	create_testcase(operator, operator_info['inputs'], operator_info['outputs'], operator_info['nodes'], declarations)
