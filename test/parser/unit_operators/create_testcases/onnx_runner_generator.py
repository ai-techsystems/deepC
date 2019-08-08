import os

def generate_onnx_runner(op_name, inputs):
	py_file = '''
# Copyright 2018 The DNNC Authors. All Rights Reserved.
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
\n\n'''

	py_file += "import onnx\n"
	py_file += "import caffe2.python.onnx.backend\n"
	py_file += "from caffe2.python import core, workspace\n"
	py_file += "import numpy as np\n\n"

	py_file += "onnx_path = '../testcases/" + op_name + "/" + op_name + ".onnx'\n"
	py_file += "modelFile = onnx.load(onnx_path)\n"
	py_file += "inputs = " + inputs + "\n"
	py_file += "output = caffe2.python.onnx.backend.run_model(modelFile, inputs)\n"
	py_file += "print(output)\n\n"

	path_name = '../testcases/' + op_name
	file_name = path_name + "/" + op_name + "_onnx_runner.py"

	if not os.path.isdir(path_name):
		os.system("mkdir -p " + path_name)
	with open(file_name, 'w') as f:
		f.write(py_file)
	os.system('python ' + file_name)