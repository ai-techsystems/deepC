#https://pytorch.org/docs/stable/torchvision/models.html

import os
import torch
import torchvision.models as models

models = {}
models['resnet18'] = {'dummy_input':'torch.randn(1, 3, 224, 224)'}

for model_name in models.keys():
	dummy_input = models[model_name]['dummy_input']
	
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
\n\n'''
	
	py_file += 'import sys\n'
	py_file += 'import onnx\n'	
	py_file += 'import torch\n'
	py_file += 'import torch.nn as nn\n'
	py_file += 'import torchvision.models as models\n\n'
	py_file += 'sys.path.append("../../../python/parser")\n'
	py_file += 'from onnx_parser import *\n\n'
	py_file += 'dummy_input = ' + dummy_input + '\n'
	py_file += 'model = models.' + model_name + '()\n\n'
	py_file += 'onnx_filename = "./testcases/' + model_name + '/' + model_name + '.onnx"\n'
	py_file += 'torch.onnx.export(model, dummy_input, onnx_filename)\n'
	py_file += 'sym_filename = "./gold_files/' + model_name + '.sym"\n'
	py_file += 'text_filename = "./testcases/' + model_name + '/' + model_name + '.txt"\n'
	py_file += 'with open(text_filename, "w") as f:\n'
	py_file += '\tmodel = onnx.load(onnx_filename)\n'
	py_file += '\tf.write(str(model.graph))\n'
	py_file += 'parse(onnx_filename, sym_filename, onnx_output_file=text_filename)'
	
	os.system("mkdir testcases/" + model_name)
	filename = 'testcases/' + model_name + '/' + model_name + '_generator.py'
	with open(filename, 'w') as f:
		f.write(py_file)
	
	os.system("python3 " + filename)
