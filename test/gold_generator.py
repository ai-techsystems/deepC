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
# Author: Ravit Sharma
# Date: July 22, 2019

#Tested for Python 3
#Add all .onnx files to a folder in the current working directory called "testcases"
#Output symbol tables will be present in folder called "output"

import os, sys
sys.path.append('../python/')
from onnx_parser import parse

testcase_dir = "./parser/testcases/"
output_dir = "./parser/gold_files/ONNXParser v1.0"

if not os.path.isdir(output_dir):
	os.system('mkdir -p ' + output_dir)

testcases = []
for r, d, f in os.walk(testcase_dir):
	for filename in f:
		path = r + '/' + filename		
		if ".onnx" in filename:
			testcases.append(path)

for onnx_testcase in testcases:
	name = onnx_testcase.split('/')[-1][:-5]
	print(name)
	output_filename = output_dir + name + ".sym" + ".gold"
	parse(onnx_testcase, output_filename)