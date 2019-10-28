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

import deepC.dnnc as dnnc
import deepC.scripts.read_onnx as read_onnx

class dnncCpp:
  """ write C++ file, given a DNNC graph. """

  def __init__ (self):
      self.deleteMe = ""

  def main(self, dc_graph, bundle_dir, cpp_file):
      print("Writing C++ file ", bundle_dir+os.path.sep+cpp_file);
      cppCode = dnnc.cppCodeGen(dc_graph, bundle_dir, cpp_file);
      cppCode.write();

def main():
  onnx_file = None
  if len(sys.argv) >= 2:
    onnx_file = sys.argv[1]

  if ( onnx_file is None ) :
    print("\nUsage: "+sys.argv[0]+ " <onnx_model_file>.onnx [bundle_dir] \n")
    exit(0)

  bundle_dir = None
  if len(sys.argv) >= 3:
    bundle_dir = sys.argv[2]
  else:
    bundle_dir = os.path.dirname(onnx_file);

  cpp_file = os.path.splitext(os.path.basename(onnx_file))[0]+'.cpp'

  parser = read_onnx.pbReader()
  dcGraph = parser.main(onnx_file, bundle_dir, optimize=False, checker=False)

  cppCodeGen = dnncCpp();
  cppFile = cppCodeGen.main(dcGraph, bundle_dir, cpp_file);

  print("INFO (ONNX): model files are ready in dir " + bundle_dir);

if __name__ == "__main__":
  main()
