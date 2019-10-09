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

#include <assert.h>
#include <codegen/cppCodeGen.h>

bool dnnc::cppCodeGen::write() {

  std::string code = "";

  for (dnnParameters param : _graph.parameters()) {
    code += write(param);
  }
  for (placeHolder term : _graph.inputs()) {
    code += write(term, true);
  }
  for (placeHolder term : _graph.outputs()) {
    code += write(term, false);
  }
  for (node &n : _graph) {
    code += write(n);
  }

  std::cout << code << "\n";
  return code.length();
}

std::pair<std::string, std::string>
dnnc::cppCodeGen::initializeData(irTypeData dtype) {
  std::string varType;  // int, float, std::vector<float> etc
  std::string initData; // = {1.3, 1.5} etc
  switch (dtype.type()) {
  case IR_DataType::INT8:
  case IR_DataType::INT16:
  case IR_DataType::INT32:
  case IR_DataType::INT64: {
    std::vector<int> values = std::vector<int>(dtype);
    for (auto el : values)
      initData = (initData.size() ? "," : "{") + std::to_string(el);
    initData = values.size() ? "}" : "";
    varType = getDNNC_IRTypeStr(dtype.type());
    varType = values.size() ? "std::vector<" + varType + ">" : varType + "\n";
    break;
  }
  case IR_DataType::UINT8:
  case IR_DataType::UINT16:
  case IR_DataType::UINT32:
  case IR_DataType::UINT64: {
    std::vector<unsigned int> values = std::vector<unsigned int>(dtype);
    for (auto el : values)
      initData = (initData.size() ? "," : "{") + std::to_string(el);
    initData = values.size() ? "}" : "";
    varType = getDNNC_IRTypeStr(dtype.type());
    varType = values.size() ? "std::vector<" + varType + ">" : varType + "\n";
    break;
  }
  case IR_DataType::FLOAT:
  case IR_DataType::FLOAT16:
  case IR_DataType::DOUBLE: {
    std::vector<float> values = std::vector<float>(dtype);
    for (auto el : values)
      initData = (initData.size() ? "," : "{") + std::to_string(el);
    initData = values.size() ? "}" : "";
    varType = getDNNC_IRTypeStr(dtype.type());
    varType = values.size() ? "std::vector<" + varType + ">" : varType + "\n";
    break;
  }
  case IR_DataType::STRING:
    varType = "std::string";
    initData = std::string(dtype);
    break;
  case IR_DataType::TENSOR_BOOL:
    // TODO:
    break;
  case IR_DataType::TENSOR_INT:
    // TODO:
    break;
  case IR_DataType::TENSOR_FLOAT:
    // TODO:
    break;
  default:
    assert(false && "irTypeData object created without type");
    break;
  }
  return std::pair<std::string, std::string>(varType, initData);
}

std::string dnnc::cppCodeGen::write(dnnParameters param) {
  std::pair<std::string, std::string> var = initializeData(param.data());
  return var.first + " " + param.name() + " = " + var.second + "\n";
}

std::string dnnc::cppCodeGen::write(placeHolder &term, bool in) {
  std::string code = "";
  return code;
}

std::string dnnc::cppCodeGen::write(node &n) {
  std::string node_name = n.name();
  if (node_name.empty()) {
    node_name = "aitsNode" + std::to_string(_graph.nodeIndex());
    n.setName(node_name); // for future occurances.
  }

  std::string code;
  // binary operators
  assert(n.inputs().size() == 2 && n.outputs().size() == 1);
  node in1Node(opInvalid);
  node in2Node(opInvalid);
  node outNode(opInvalid);
  if (_graph.findNodeByName(n.outputs()[0], outNode) == false ||
      _graph.findNodeByName(n.inputs()[0], in1Node) == false ||
      _graph.findNodeByName(n.inputs()[1], in2Node) == false) {
    ; // return code;
  }

  std::string outType = "float"; // TODO: levelize graph, infer out types.
                                 // getDNNC_IRTypeStr(outNode.type());
  std::string in1Type = "float", in2Type = "float";
  code += "MatMul<" + outType + "> " + node_name + "(\"" + node_name + "\");\n";
  for (nodeAttribute attr : n) {
    std::string name = getAttrNameStr(attr.name());
    std::pair<std::string, std::string> var = initializeData(attr.data());
    code += var.first + " " + name + " = " + var.second;
    code += node_name + ".setAttribute( attr_" + name + ", " + name + ");";
  }
  code += "tensor<" + outType + "> " + node_name + "_" + n.outputs()[0] +
          " = " + node_name + ".compute(" + node_name + "_" + n.inputs()[0] +
          "," + node_name + "_" + n.inputs()[1] + ");";

  return code;
}
