#Copyright 2018 The AITS DNNC Authors. All Rights Reserved.
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

import os

def create_testcase (op_name, inputs, outputs, nodes, declarations=[]):
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
	
	py_file += 'import os, sys\n\n'
	py_file += 'from onnx import *\n'
	py_file += 'sys.path.append(".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "python" + os.sep + "parser")\n'
	py_file += 'from onnx_parser import *\n\n'
	py_file += 'op_name = \'' + op_name + '\'\n\n'
	py_file += 'inputs = ' + inputs + '\n'
	py_file += 'outputs = ' + outputs + '\n'
	for declaration in declarations:
		py_file += declaration + '\n'
	py_file += 'nodes = []\n'
	for node in nodes:
		py_file+='nodes.append(' + node + ')\n'
	py_file += 'graph = helper.make_graph(nodes, op_name+"_graph", inputs, outputs)\n'
	py_file += 'opset = (OperatorSetIdProto(version=11),)\n'
	py_file += 'model = helper.make_model(graph, opset_imports=opset)\n'
	py_file += 'onnx.checker.check_model(model)\n'
	py_file += 't_prefix = ".." + os.sep + "testcases" + os.sep  + op_name + os.sep + op_name\n'
	py_file += 'g_prefix = ".." + os.sep + "gold_files" + os.sep + op_name\n'
	py_file += 'onnx.save(model, t_prefix+".onnx")\n'
	py_file += 'parse(t_prefix+".onnx", g_prefix+".sym", onnx_output_file=t_prefix+".txt")\n'
	
	if not os.path.isdir(op_name):
		command = "mkdir -p " + ".." + os.sep + "testcases" + os.sep + op_name
		os.system(command)
	file_name = ".." + os.sep + "testcases" + os.sep + op_name + os.sep + op_name + "_generator.py"
	with open(file_name, 'w') as f:
		f.write(py_file)
		
	os.system('python ' + file_name)
