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
import deepC.dnnc as dnnc
import deepC.scripts.onnx2cpp as onnx2cpp

class compilerWrapper:

  """Compiler class for models in ONNX binary/protobuf format."""

  def __init__ (self):
    dnnc_path = os.path.abspath(os.path.dirname(dnnc.__file__))
    self.inc_path = "-I " + os.path.join(dnnc_path, "include")
    self.isys_path = "-isystem " +  os.path.join(dnnc_path, "packages", "eigen-eigen-323c052e1731")
    self.compiler = "g++"
    self.cpp_flags = "-O3"

  def cmd (self, cppFile, exeFile):
    return ' '.join([self.compiler,
        self.cpp_flags,
        self.inc_path,
        self.isys_path,
        cppFile,
        '-o', exeFile])

  def compile(self, cppFile):
    from subprocess import PIPE, run
    exeFile = os.path.splitext(cppFile)[0]+".exe"
    command = self.cmd(cppFile, exeFile);
    print(command)
    compileProcess = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)

    sys.stdout.write(compileProcess.stdout)
    if ( compileProcess.returncode ):
      sys.stderr.write(compileProcess.stderr)
      sys.stderr.write("\ndnnc compilation failed. please file this bug with model/script file at\n");
      sys.stderr.write("    https://github.com/ai-techsystems/dnnCompiler/issues\n");
      return None;

    return exeFile

# Use Model:
#    _bundleDir        : dirname("generated exe, i.e. a.out");
#    parameter file(s) : in _bundleDir
#    input     file(s) : with a path relative to current dir.
#    output    file(s) : in current dir
def main():
  onnx_file = None
  if len(sys.argv) >= 2:
    onnx_file = sys.argv[1]

  compile_flags = None
  if len(sys.argv) >= 4:
    compile_flags = sys.argv[3]
    sys.argv[3] = None

  if ( onnx_file is None ) :
    print("\nUsage: "+sys.argv[0]+ " <onnx_model_file>.onnx [bundle_dir] [compile_flags] \n")
    exit(0)

  (bundleDir, cppFile) = onnx2cpp.main();

  onnxCC = compilerWrapper();
  exe = onnxCC.compile(os.path.join(bundleDir, cppFile));

  if ( exe is not None and exe ):
    print("model executable ", exe);
  else:
    print("\nUsage: "+sys.argv[0]+ " <onnx_model_file>.onnx \n")

if __name__ == "__main__":
  sys.exit(main())
