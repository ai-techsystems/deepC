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
#
############################
# Description:
#   DNNC CPP FIle generator
#############################

import os, sys
if __name__ == "__main__":
  DNNC_PATH=os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
  sys.path.append(DNNC_PATH+os.path.sep+'swig')
  sys.path.append(DNNC_PATH+os.path.sep+'python')

import dnnc
import read_onnx

class dnncCpp:
  """ write C++ file, given a DNNC graph. """

  def __init__ (self):
      self.deleteMe = ""

  def main(self, dc_graph, cpp_file=""):
      print("Writing C++ file", cpp_file);
      cppCode = dnnc.cppCodeGen(dc_graph, cpp_file);
      cppCode.write();

if __name__ == "__main__":
  if len(sys.argv) >= 2:
    onnx_file = sys.argv[1];
    parser = read_onnx.pbReader()
    dcGraph = parser.main(sys.argv[1], optimize=False, checker=False)

    if ( len(sys.argv) >=3 ):
        cpp_file = sys.argv[2];
    else:
        cpp_file = os.path.splitext(os.path.abspath(onnx_file))[0]+'.cpp'
    cppCodeGen = dnncCpp();
    cppFile = cppCodeGen.main(dcGraph, cpp_file);

  else:
    print("\nUsage: "+sys.argv[0]+ " <onnx_model_file>.onnx \n")
