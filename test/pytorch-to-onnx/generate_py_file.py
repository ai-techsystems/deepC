def generate_py_file(operator_name, output_string, test_input_string, init=None, is_module=False):
  #operator_name --> onnx_filename, text_filename, class name
  #input_string --> test_input
  py_file_name = './' + operator_name + '/' + operator_name + '_generator.py'
  
  onnx_filename = './' + operator_name + '/' + operator_name + '.onnx'
  text_filename = './' + operator_name + '/' + operator_name + '.txt'
  
  py_file = '''
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
	# Author:
	# Date:\n\n'''
  
  py_file += 'import torch.onnx\n'
  py_file += 'import torch.nn as nn\n'
  py_file += 'import numpy as np\n'
  py_file += 'import onnx\n\n'
  
  py_file += 'onnx_filename = "' + onnx_filename + '"\n'
  py_file += 'text_filename = "' + text_filename + '"\n\n'
  
  if is_module:
    py_file += 'model = ' + output_string + '\n'
  else:
    py_file += 'class Net(nn.Module):\n'
    py_file += '\tdef __init__(self):\n'
    py_file += '\t\tsuper(Net, self).__init__()\n'
    for init_string in init:
      py_file += '\t\t' + init_string + '\n'
    py_file += '\tdef forward(self, x):\n'
    py_file += '\t\toutput = ' + output_string + '\n'
    py_file += '\t\treturn output\n\n'

    py_file += 'model = Net()\n'
  
  py_file += 'test_input = ' + test_input_string + '\n'
  py_file += 'torch.onnx.export(model, test_input, onnx_filename)\n'
  
  py_file += "with open(text_filename, 'w') as f:\n"
  py_file += '\tmodel = onnx.load(onnx_filename)\n'
  py_file += '\tf.write(str(model.graph))'
  
  with open (py_file_name, 'w') as f:
    f.write(py_file)
  
  return py_file_name
