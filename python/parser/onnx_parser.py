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
# Author: Colin Lu
# Date: July 18, 2019

import onnx
import sys

def get_node_symbol(node):
	symbol = ""
	symbol += node.op_type + '\n'
	symbol += '\t"' + node.name + '"\n'
	
	symbol += '\t[ '
	for elem in node.input:
		symbol += '[\"' + elem + '\"] '
	symbol += ']\n'

	symbol += '\t[ '
	for elem in node.output:
		symbol += '[\"' + elem + '\"] '
	symbol += ']\n'

	for attr in node.attribute:
		symbol += '\tAttr\n'
		symbol += '\t\t"' + node.name + '/' + attr.name + '"\n'
		symbol += '\t\t[ '

		if attr.type == onnx.AttributeProto.INT:
			symbol += str(attr.i)
		elif attr.type == onnx.AttributeProto.INTS:
			symbol += '['
			for index, num in enumerate(attr.ints):
				symbol += str(num)
				if index != len(attr.ints) - 1:
					symbol += ', '
			symbol += ']'
		elif attr.type == onnx.AttributeProto.FLOAT:
			symbol += str(attr.f)
		elif attr.type == onnx.AttributeProto.FLOATS:
			symbol += '['
			for index, num in enumerate(attr.floats):
				symbol += str(num)
				if index != len(attr.floats) - 1:
					symbol += ', '
			symbol += ']'
		symbol += ' ]\n'

	symbol += '\n'
	return symbol

def get_io_symbol(io, is_input):
	symbol = ""

	if is_input:
		symbol += "Input"
	else:
		symbol += "Output"
	symbol += '\n'
	symbol += '\t"' + io.name + '"\n'

	symbol += '\t[ '
	dims = io.type.tensor_type.shape.dim
	for i, dim in enumerate(dims):
		if dim.dim_param != "":
			symbol += dim.dim_param
		else:
			symbol += str(dim.dim_value)
		if i != len(dims) - 1:
			symbol += ', '
	symbol += ' ]\n'

	symbol += '\n'
	return symbol

def get_initializer_symbol(initializer):
	symbol = "Initializer\n"
	symbol += '\t"' + initializer.name + '"\n'

	symbol += '\t[ '
	for i, dim in enumerate(initializer.dims):
		symbol += str(dim)
		if i != len(initializer.dims) - 1:
			symbol += ', '
	symbol += ' ]\n'

	float_data_types = [1, 14] # data_type values which store data in the float_data field (including complex)
	int32_data_types = [2, 3, 4, 5, 6, 9, 10] # data_type values which store data in int32_data
	int64_data_types = [7] # data_type values which store data in int64_data
	
	if initializer.raw_data != b'': # if the raw_data field is not empty
		symbol += '\t[ ' + str(initializer.raw_data) + ' ]\n'
	elif initializer.data_type in float_data_types:
		symbol += '\t[ ' + str(initializer.float_data) + ' ]\n'
	elif initializer.data_type in int32_data_types:
		symbol += '\t[ ' + str(initializer.int32_data) + ' ]\n'
	elif initializer.data_type in int64_data_types:
		symbol += '\t[ ' + str(initializer.int64_data) + ' ]\n'

	symbol += '\n'
	return symbol

def get_symbol_table(onnx_filename):
	model = onnx.load(onnx_filename)

	graph = model.graph
	nodes = graph.node

	output = ""
	
	for node in nodes:
		output += get_node_symbol(node)

	for i in graph.input:
		output += get_io_symbol(i, True)

	for o in graph.output:
		output += get_io_symbol(o, False)

	for initl in graph.initializer:
		output += get_initializer_symbol(initl)

	return output


def parse(onnx_filename, output_file, onnx_output_file=None):
	print("writing to", output_file)
	with open(output_file, "w") as f:
		f.write(get_symbol_table(onnx_filename))

	if onnx_output_file is not None:
		model = onnx.load(onnx_filename)
		with open(onnx_output_file, "w") as f:
			f.write(str(model))

if __name__ == "__main__":
	if len(sys.argv) >= 4:
		parse(sys.argv[1], sys.argv[2], sys.argv[3])
	else:
		parse(sys.argv[1], sys.argv[2])