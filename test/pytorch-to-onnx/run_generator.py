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

import automate

operators = {}

# simple op, single-tensor
operators['sigmoid'] = {'output_string':'torch.sigmoid(x)', 'test_input_string':'torch.Tensor([1,2,3])'}
operators['mean'] = {'output_string':'torch.mean(x)', 'test_input_string':'torch.randn(1, 5)'}
operators['max'] = {'output_string': 'torch.max(x)', 'test_input_string': 'torch.full([3, 3], 4)'}
operators['exp'] = {'output_string': 'torch.exp(x)', 'test_input_string': 'torch.full([3, 3], 4)'}

# simple op, multiple-tensor
operators['add'] = {'output_string':'torch.add(x, self.to_add)', 'test_input_string':'torch.Tensor([3,3,3])', 'init':['self.to_add = torch.Tensor([2,2,2])']}
operators['cat'] = {'output_string':'torch.cat([x, self.to_cat])', 'test_input_string':'torch.Tensor([[1,2],[3,4]])', 'init':['self.to_cat = torch.Tensor([[5,6],[7,8]])']}
operators['eq'] = {'output_string': 'torch.eq(x, self.to_compare)', 'test_input_string': 'torch.full([3, 3], 4)', 'init': ['self.to_compare = torch.full([3, 3], 2)']}
operators['gt'] = {'output_string': 'torch.gt(x, self.to_compare)', 'test_input_string': 'torch.full([3, 3], 4)', 'init': ['self.to_compare = torch.full([3, 3], 2)']}

# torch.nn.module ops
operators['PReLU'] = {'output_string':'nn.PReLU()', 'test_input_string':'torch.randn(2)', 'is_module':True}
operators['threshold'] = {'output_string':'nn.Threshold(0, 0)', 'test_input_string':'torch.randn(2)', 'is_module':True}
operators['softmax'] = {'output_string':'nn.Softmax()', 'test_input_string':'torch.randn(6)', 'is_module':True}
operators['elu'] = {'output_string':'nn.ELU()', 'test_input_string':'torch.randn(10)', 'is_module':True}
operators['Conv'] = {'output_string': 'torch.nn.Conv2d(4, 5, 3)', 'test_input_string': 'torch.full([1, 4, 5, 6], 4)', 'is_module': True}
operators['BatchNorm'] = {'output_string': 'torch.nn.BatchNorm1d(6)', 'test_input_string': 'torch.full([3, 6], 4)', 'is_module': True}
operators['MaxPool2d'] = {'output_string': 'torch.nn.MaxPool2d(3, stride=1)', 'test_input_string': 'torch.full([1, 4, 5, 5], 4)', 'is_module': True}
operators['Embedding'] = {'output_string': 'torch.nn.Embedding(10, 3)', 'test_input_string': 'torch.LongTensor([[1,2,4,5],[4,3,2,9]])', 'is_module': True}
operators['RNN'] = {'output_string': 'torch.nn.RNN(4, 6, 3)', 'test_input_string': 'torch.randn(10, 1, 4)', 'is_module': True}
operators['ConstantPad2d'] = {'output_string': 'torch.nn.ConstantPad2d(2, 4.5)', 'test_input_string': 'torch.full([1, 3, 4, 4], 3)', 'is_module': True}
operators['Dropout'] = {'output_string': 'torch.nn.Dropout(p=0.3)', 'test_input_string': 'torch.full([4, 4], 3)', 'is_module': True}


import os
for operator in operators.keys():
	if not os.path.isdir(operator):
		os.system("mkdir " + operator)

	output_string = operators[operator]['output_string']
	test_input_string = operators[operator]['test_input_string']
	
	init = []
	if 'init' in operators[operator]:
		init = operators[operator]['init']

	if 'is_module' in operators[operator]:
		is_module = operators[operator]['is_module']
		file = automate.generate_py_file(operator, output_string, test_input_string, init, is_module)
	else:
		file = automate.generate_py_file(operator, output_string, test_input_string, init)
	os.system("python " + file)