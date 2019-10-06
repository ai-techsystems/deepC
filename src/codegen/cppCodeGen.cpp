// Copyright 2018 The AITS DNNC Authors.All Rights Reserved.
//
// Licensed to the Apache Software Foundation(ASF) under one
// or more contributor license agreements.See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.See the License for the
// specific language governing permissionsand limitations
// under the License.
//
// This file is part of AITS DNN compiler maintained at
// https://github.com/ai-techsystems/dnnCompiler
//

#include <codegen/cppCodeGen.h>

bool dnnc::cppCodeGen::write() {

  bool result = true;

  for (dnnParameters param : _graph.parameters()) {
    write(param.data(), param.name());
  }
  for (placeHolder term : _graph.inputs()) {
    write(term, true);
  }
  for (placeHolder term : _graph.outputs()) {
    write(term, false);
  }
  for (node &n : _graph) {
    result &= write(n);
  }
  return result;
}

bool dnnc::cppCodeGen::write(irTypeData param, std::string name) {
  bool result = true;
  return result;
}

bool dnnc::cppCodeGen::write(placeHolder &term, bool in) {
  bool result = true;
  return result;
}

bool dnnc::cppCodeGen::write(node &n) {
  bool result = true;
  for (nodeAttribute attr : n) {
    write(attr.data(), getAttrNameStr(attr.name()));
  }
  return result;
}
