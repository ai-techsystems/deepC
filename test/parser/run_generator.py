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
#
# Author: Ravit Sharma, Colin Lu
# Date: July 25, 2019

'''
Example for using the generator file, turning PyTorch ops into an ONNX graph.
Note that the generator creates a new subfolder for each operator provided.
Each operator requires an output_string and test_input_string, other arguments depend on the operator used.
Example ops are provided, although they are not the complete set of PyTorch ops supported by ONNX.
'''

import generator
import os

operators = {}
operators['acos'] = {'output_string':'torch.acos(x)', 'test_input_string':'torch.randn(1, 5)'}
operators['add'] = {'output_string':'torch.add(x, self.to_add)', 'test_input_string':'torch.Tensor([3,3,3])', 'init':['self.to_add = torch.Tensor([2,2,2])']}
operators['sub'] = {'output_string':'torch.sub(x, self.to_sub)', 'test_input_string':'torch.Tensor([3,3,3])', 'init':['self.to_sub = torch.Tensor([2,2,2])']}
operators['mul'] = {'output_string':'torch.mul(x, self.to_mul)', 'test_input_string':'torch.Tensor([3,3,3])', 'init':['self.to_mul = torch.Tensor([2,2,2])']}
operators['div'] = {'output_string':'torch.div(x, self.to_div)', 'test_input_string':'torch.Tensor([3,3,3])', 'init':['self.to_div = torch.Tensor([2,2,2])']}
operators['cat'] = {'output_string':'torch.cat([x, self.to_cat])', 'test_input_string':'torch.Tensor([[1,2],[3,4]])', 'init':['self.to_cat = torch.Tensor([[5,6],[7,8]])']}
operators['mm'] = {'output_string':'torch.mm(x, self.to_mm)', 'test_input_string':'torch.randn(1,3)', 'init':['self.to_mm = torch.randn(3,2)']}
operators['addmm'] = {'output_string':'torch.addmm(x, self.mat1, self.mat2)', 'test_input_string':'torch.randn(3,3)', 'init':['self.mat1 = torch.randn(3,2)', 'self.mat2 = torch.randn(2,3)']}
operators['neg'] = {'output_string':'torch.neg(x)', 'test_input_string':'torch.Tensor([1,2,3])'}
operators['sqrt'] = {'output_string':'torch.sqrt(x)', 'test_input_string':'torch.Tensor([1,2,3])'}
operators['tanh'] = {'output_string':'torch.tanh(x)', 'test_input_string':'torch.Tensor([1,2,3])'}
operators['sigmoid'] = {'output_string':'torch.sigmoid(x)', 'test_input_string':'torch.Tensor([1,2,3])'}
operators['mean'] = {'output_string':'torch.mean(x)', 'test_input_string':'torch.randn(1, 5)'}
operators['sum'] = {'output_string':'torch.sum(x)', 'test_input_string':'torch.randn(1,5)'}
operators['prod'] = {'output_string':'torch.prod(x)', 'test_input_string':'torch.randn(1, 5)'}
operators['t'] = {'output_string':'torch.t(x)', 'test_input_string':'torch.randn(2,3)'}
#doublecheck on expand
#operators['expand'] = {'output_string':'torch.add(x, self.to_add)', 'test_input_string':'torch.randn(2,3)', 'init':['self.to_add = torch.randn(1,6)']}
operators['transpose'] = {'output_string':'torch.transpose(x, 1, 2)', 'test_input_string':'torch.randn(1,2,3)'}
operators['view'] = {'output_string':'x.view(x.size(0), -1)', 'test_input_string':'torch.randn(2,3)'}
operators['split'] = {'output_string':'torch.split(x, 5)', 'test_input_string':'torch.randn(10,3)'}
operators['squeeze'] = {'output_string':'torch.squeeze(x)', 'test_input_string':'torch.randn(1,5)'}
operators['PReLU'] = {'output_string':'nn.PReLU()', 'test_input_string':'torch.randn(2)', 'is_module':True}
#for some reason, threshold seems to be only supported for threshold=0 and value=0
operators['threshold'] = {'output_string':'nn.Threshold(0, 0)', 'test_input_string':'torch.randn(2)', 'is_module':True}
operators['leaky_relu'] = {'output_string':'nn.LeakyReLU()', 'test_input_string':'torch.randn(2)', 'is_module':True}
operators['glu'] = {'output_string':'nn.functional.glu(x)', 'test_input_string':'torch.randn(10)'}
operators['softmax'] = {'output_string':'nn.Softmax()', 'test_input_string':'torch.randn(6)', 'is_module':True}
operators['avg_pool2D'] = {'output_string':'nn.functional.avg_pool2d(x, self.kernel_size)', 'test_input_string':'torch.randn(32,3,28,28)', 'init':['self.kernel_size = (5,5)']}
operators['log_softmax'] = {'output_string':'nn.functional.log_softmax(x)', 'test_input_string':'torch.randn(10)'}
#cannot export unfold
#operators['unfold'] = {'output_string':'nn.functional.unfold(x, self.kernel_size)', 'test_input_string':'torch.randn(32,3,28,28)', 'init':['self.kernel_size = (5,5)']}
operators['elu'] = {'output_string':'nn.ELU()', 'test_input_string':'torch.randn(10)', 'is_module':True}
operators['concat'] = {'output_string':'torch.cat((x, self.to_cat))', 'test_input_string':'torch.randn(5,10)', 'init':['self.to_cat = torch.randn(2,10)']}
operators['abs'] = {'output_string':'torch.abs(x)', 'test_input_string':'torch.Tensor([-1,-2,2])'}
operators['index_select'] = {'output_string':'torch.index_select(x, 0, self.indices)', 'test_input_string':'torch.randn(5,5)', 'init':['self.indices = torch.tensor([0,2,3])']}
operators['pow'] = {'output_string':'torch.pow(x, self.exp)', 'test_input_string':'torch.rand(5)', 'init':['self.exp = 2']}
operators['clamp'] = {'output_string':'torch.clamp(x, self.min, self.max)', 'test_input_string':'torch.randn(5)', 'init':['self.min = 0.25', 'self.max = 0.75']}

operators['max'] = {'output_string': 'torch.max(x)', 'test_input_string': 'torch.full([3, 3], 4)'}
operators['min'] = {'output_string': 'torch.min(x)', 'test_input_string': 'torch.full([3, 3], 4)'}
operators['exp'] = {'output_string': 'torch.exp(x)', 'test_input_string': 'torch.full([3, 3], 4)'}
operators['sin'] = {'output_string': 'torch.sin(x)', 'test_input_string': 'torch.full([3, 3], 4)'}
operators['cos'] = {'output_string': 'torch.cos(x)', 'test_input_string': 'torch.full([3, 3], 4)'}
operators['tan'] = {'output_string': 'torch.tan(x)', 'test_input_string': 'torch.full([3, 3], 4)'}
operators['asin'] = {'output_string': 'torch.asin(x)', 'test_input_string': 'torch.full([3, 3], 4)'}
operators['acos'] = {'output_string': 'torch.acos(x)', 'test_input_string': 'torch.full([3, 3], 4)'}
operators['atan'] = {'output_string': 'torch.atan(x)', 'test_input_string': 'torch.full([3, 3], 4)'}
operators['permute'] = {'output_string': 'x.permute(1, 0)', 'test_input_string': 'torch.Tensor([[1, 2, 3], [4, 5, 6]])'}

# simple op, comparison (two-tensor)
operators['eq'] = {'output_string': 'torch.eq(x, self.to_compare)', 'test_input_string': 'torch.full([3, 3], 4)', 'init': ['self.to_compare = torch.full([3, 3], 2)']}
operators['gt'] = {'output_string': 'torch.gt(x, self.to_compare)', 'test_input_string': 'torch.full([3, 3], 4)', 'init': ['self.to_compare = torch.full([3, 3], 2)']}
operators['lt'] = {'output_string': 'torch.lt(x, self.to_compare)', 'test_input_string': 'torch.full([3, 3], 4)', 'init': ['self.to_compare = torch.full([3, 3], 2)']}
operators['ge'] = {'output_string': 'torch.ge(x, self.to_compare)', 'test_input_string': 'torch.full([3, 3], 4)', 'init': ['self.to_compare = torch.full([3, 3], 2)']}
operators['le'] = {'output_string': 'torch.le(x, self.to_compare)', 'test_input_string': 'torch.full([3, 3], 4)', 'init': ['self.to_compare = torch.full([3, 3], 2)']}

# torch.nn.module ops
operators['Conv'] = {'output_string': 'torch.nn.Conv2d(4, 5, 3)', 'test_input_string': 'torch.full([1, 4, 5, 6], 4)', 'is_module': True}
operators['BatchNorm'] = {'output_string': 'torch.nn.BatchNorm1d(6)', 'test_input_string': 'torch.full([3, 6], 4)', 'is_module': True}
operators['MaxPool1d'] = {'output_string': 'torch.nn.MaxPool1d(3, stride=1)', 'test_input_string': 'torch.full([1, 4, 5], 4)', 'is_module': True}
operators['MaxPool2d'] = {'output_string': 'torch.nn.MaxPool2d(3, stride=1)', 'test_input_string': 'torch.full([1, 4, 5, 5], 4)', 'is_module': True}
operators['MaxPool3d'] = {'output_string': 'torch.nn.MaxPool3d(3, stride=1)', 'test_input_string': 'torch.full([1, 4, 5, 5, 5], 4)', 'is_module': True}
operators['Embedding'] = {'output_string': 'torch.nn.Embedding(10, 3)', 'test_input_string': 'torch.LongTensor([[1,2,4,5],[4,3,2,9]])', 'is_module': True}
operators['RNN'] = {'output_string': 'torch.nn.RNN(4, 6, 3)', 'test_input_string': 'torch.randn(10, 1, 4)', 'is_module': True}
operators['ConstantPad1d'] = {'output_string': 'torch.nn.ConstantPad1d(2, 4.5)', 'test_input_string': 'torch.full([1, 3, 4], 3)', 'is_module': True}
operators['ConstantPad2d'] = {'output_string': 'torch.nn.ConstantPad2d(2, 4.5)', 'test_input_string': 'torch.full([1, 3, 4, 4], 3)', 'is_module': True}
operators['ConstantPad3d'] = {'output_string': 'torch.nn.ConstantPad3d(2, 4.5)', 'test_input_string': 'torch.full([1, 3, 4, 4, 4], 3)', 'is_module': True}
operators['Dropout'] = {'output_string': 'torch.nn.Dropout(p=0.3)', 'test_input_string': 'torch.full([4, 4], 3)', 'is_module': True}


for operator in operators.keys():
	output_string = operators[operator]['output_string']
	test_input_string = operators[operator]['test_input_string']
	
	init = []
	if 'init' in operators[operator]:
		init = operators[operator]['init']

	if 'is_module' in operators[operator]:
		is_module = operators[operator]['is_module']
		file = onnx_generator.generate_py_file(operator, output_string, test_input_string, init=init, is_module=is_module)
	else:
		file = onnx_generator.generate_py_file(operator, output_string, test_input_string, init=init)
	os.system("python " + file)
