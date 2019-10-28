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

############################
# Description:
#   DNNC AOT Compiler script
#############################

import os, sys
if __name__ == "__main__":
  DNNC_PATH=os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..'+os.path.sep+'swig')
  sys.path.append(DNNC_PATH)

import dnnc
import read_onnx
import onnx_cpp

def compilerWrapper;
  """Compiler class for models in ONNX binary/protobuf format."""

def main():
  if len(sys.argv) >= 2:
    parser = pbReader()
    dcGraph = parser.main(sys.argv[1])

    cppCodeGen = dnncCpp();
    cppFile = cppCodeGen.main(gcGraph);

    onnxCC = compilerWrapper();
    onnxCC.main(cppFile);
  else:
    print("\nUsage: "+sys.argv[0]+ " <onnx_model_file>.onnx \n")

if __name__ == "__main__":
  main()
