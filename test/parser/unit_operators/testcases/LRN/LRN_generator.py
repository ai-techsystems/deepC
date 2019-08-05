
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
from onnx_parser import *

op_name = 'LRN'

inputs = [helper.make_tensor_value_info("x",TensorProto.FLOAT,(2,3,4,5))]
outputs = [helper.make_tensor_value_info("y",TensorProto.FLOAT,(2,3,4,5))]
nodes = []
nodes.append(helper.make_node("LRN",["x"],["y"],alpha = 0.0001,beta = 0.75,bias = 1.0,size = 3))
graph = helper.make_graph(nodes, op_name+"_graph", inputs, outputs)
opset = (OperatorSetIdProto(version=11),)
model = helper.make_model(graph, opset_imports=opset)
onnx.checker.check_model(model)
t_prefix = ".." + separator + "testcases" + separator + op_name + separator + op_name
g_prefix = ".." + separator + "gold_files" + separator + op_name
onnx.save(model, t_prefix+".onnx")
parse(t_prefix+".onnx", g_prefix+".sym", onnx_output_file=t_prefix+".txt")
