
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


import os, sys

import numpy as np
separator = os.path.sep

from onnx import *
sys.path.append(".."+separator+".."+separator+".."+separator+".."+separator+"python/parser")
# from onnx_parser import *

op_name = 'LSTM'

seq_length = 3
batch_size = 3
input_size = 4
hidden_size = 3
num_directions = 1
weight_scale = 0.1
number_of_gates = 4
number_of_peepholes = 3

inputs = [helper.make_tensor_value_info('X',TensorProto.FLOAT,[seq_length, batch_size, input_size]),
    helper.make_tensor_value_info('W',TensorProto.FLOAT,[num_directions, number_of_gates*hidden_size, input_size]),
    helper.make_tensor_value_info('R',TensorProto.FLOAT,[num_directions, number_of_gates*hidden_size, hidden_size]),
    helper.make_tensor_value_info('B',TensorProto.FLOAT,[num_directions, 2*number_of_gates*hidden_size]),
    helper.make_tensor_value_info('sequence_lens',TensorProto.FLOAT,[batch_size]),
    helper.make_tensor_value_info('initial_h',TensorProto.FLOAT,[num_directions, batch_size, hidden_size]),
    helper.make_tensor_value_info('initial_c',TensorProto.FLOAT,[num_directions, batch_size, hidden_size]),
    helper.make_tensor_value_info('P',TensorProto.FLOAT,[num_directions, 3*hidden_size])]


outputs = [helper.make_tensor_value_info('Y',TensorProto.FLOAT,[seq_length, num_directions, batch_size, hidden_size])]
nodes = []
nodes.append(helper.make_node('LSTM',inputs=['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c', 'P'], outputs=['Y'],hidden_size=3))
graph = helper.make_graph(nodes, op_name+"_graph", inputs, outputs)
opset = (OperatorSetIdProto(version=7),)
model = helper.make_model(graph, opset_imports=opset)
onnx.checker.check_model(model)
t_prefix = ".." + separator + separator + op_name + separator + op_name
g_prefix = ".." + separator + "gold_files" + separator + op_name
onnx.save(model, t_prefix+".onnx")
# parse(t_prefix+".onnx", g_prefix+".sym", onnx_output_file=t_prefix+".txt")
