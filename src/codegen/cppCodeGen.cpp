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
#include <fstream>

bool dnnc::cppCodeGen::write() {

  _includes.clear();

  std::string code = "";

  for (dnnParameters param : _graph.parameters()) {
    code += write(param);
  }
  for (ioNode *term : _graph.inputs()) {
    code += write(*term);
  }

  for (node *n : _graph) {
    if (n->ntype() == node::OPERATOR)
      code += write(*dynamic_cast<opNode *>(n));
  }

  // OUTPUTs are written with operators.

  std::ofstream out(_outFile);
  if (out.fail()) {
    std::cerr << "ERROR (CODEGEN): could not open file " + _outFile +
                     "to write.\n";
    return false;
  }
  out << writeIncludes() << "\n";
  out << writeMainFunction(code) << "\n";
  out.close();
  return code.length();
}

std::string dnnc::cppCodeGen::nodeName(node *n) {
  return "dnnc__node_" + n->name();
}

std::string dnnc::cppCodeGen::writeIncludes() {
  std::string code;
  for (auto &s : _includes)
    code += std::string("#include \"") + s + "\"\n";
  return code;
}

std::string dnnc::cppCodeGen::writeMainFunction(std::string body) {
  std::string code = "using namespace dnnc;\n\n";
  code += "int main() {\n\n";
  code += body + "\n";
  code += _tab + "return 0;\n";
  code += "}\n";
  return code;
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
      initData += (initData.size() ? "," : "{") + std::to_string(el);
    initData += values.size() ? "}" : "";
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
      initData += (initData.size() ? "," : "{") + std::to_string(el);
    initData += values.size() ? "}" : "";
    varType = getDNNC_IRTypeStr(dtype.type());
    varType = values.size() ? "std::vector<" + varType + ">" : varType + "\n";
    break;
  }
  case IR_DataType::FLOAT:
  case IR_DataType::FLOAT16:
  case IR_DataType::DOUBLE: {
    std::vector<float> values = std::vector<float>(dtype);
    for (auto el : values)
      initData += (initData.size() ? "," : "{") + std::to_string(el);
    initData += values.size() ? "}" : "";
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
  return var.first + " " + param.name() + " = " + var.second + " ;\n";
}

std::string dnnc::cppCodeGen::write(ioNode &term) {
  std::string dtype = getDNNC_DataTypeStr(term.dtype());
  std::string shape;
  for (size_t i : term.shape())
    shape += shape.size() ? ", " : "" + std::to_string(i);
  return _tab + "tensor<" + dtype + "> " + nodeName(&term) + " ;\n";
}

std::string dnnc::cppCodeGen::write(opNode &computeNode) {

  std::string code;

  assert(computeNode.ntype() == node::OPERATOR);

  assert(computeNode.symbol() != opInvalid);

  std::string opCode = getOpCodeStr(computeNode.symbol());
  _includes.push_back("operators/" + opCode + ".h");

  std::string opName = computeNode.name();

  assert(opName.length());

  std::vector<node *> ins, outs;
  if (false == computeNode.inputNodes(_graph, ins) ||
      false == computeNode.outputNodes(_graph, outs)) {
    std::cerr
        << "ERROR (CODEGEN): cound not find all nodes for " << opName << ",\n"
        << "                 an instance of " << opCode << ".\n"
        << "                 Please check model's sanity and try again.\n";
    return code;
  }

  if (ins.size() == 1 && outs.size() == 1) {
    code = writeUnaryOperator(computeNode, ins, outs);
  } else if (ins.size() == 2 && outs.size() == 1) {
    code = writeBinaryOperator(computeNode, ins, outs);
  } else if (ins.size() == 3 && outs.size() == 1) {
    code = writeTernaryOperator(computeNode, ins, outs);
  } else {
    code = writeCustomOperator(computeNode, ins, outs);
  }
  return code + "\n";
}

std::string dnnc::cppCodeGen::writeUnaryOperator(opNode &computeNode,
                                                 std::vector<node *> &ins,
                                                 std::vector<node *> &outs) {
  std::string code;

  assert(ins.size() == 1 && outs.size() == 1);

  std::string opCode = getOpCodeStr(computeNode.symbol());

  std::string opName = computeNode.name();

  assert(opName.length());

  std::string outType = getDNNC_DataTypeStr(outs[0]->dtype());
  std::string inType = getDNNC_DataTypeStr(ins[0]->dtype());

  // Step 1: Instantiate opterator
  code += _tab + opCode + "<" + outType + ", " + inType + ", " + inType + "> " +
          opName + "(\"" + opName + "\");\n";

  // Step 2: Add attribute
  for (nodeAttribute attr : computeNode) {
    std::string attrName = getAttrNameStr(attr.name());
    std::pair<std::string, std::string> var = initializeData(attr.data());
    code += _tab + var.first + " " + attrName + " = " + var.second + " ;\n";
    code += _tab + opName + ".setAttribute ( attr_" + attrName + ", " +
            attrName + " );\n";
  }

  // Step 3: Add compute function.
  code += _tab + "tensor<" + outType + "> " + nodeName(outs[0]) + " = " +
          opName + ".compute ( " + nodeName(ins[0]) + ");\n";

  return code;
}

std::string dnnc::cppCodeGen::writeBinaryOperator(opNode &computeNode,
                                                  std::vector<node *> &ins,
                                                  std::vector<node *> &outs) {
  std::string code;

  assert(ins.size() == 2 && outs.size() == 1);

  std::string opCode = getOpCodeStr(computeNode.symbol());

  std::string opName = computeNode.name();

  assert(opName.length());

  std::string outType = getDNNC_DataTypeStr(outs[0]->dtype());
  std::string in1Type = getDNNC_DataTypeStr(ins[0]->dtype());
  std::string in2Type = getDNNC_DataTypeStr(ins[1]->dtype());

  // Step 1: Instantiate opterator
  code += _tab + opCode + "<" + outType + ", " + in1Type + ", " + in2Type +
          "> " + opName + "(\"" + opName + "\");\n";

  // Step 2: Add attribute
  for (nodeAttribute attr : computeNode) {
    std::string attrName = getAttrNameStr(attr.name());
    std::pair<std::string, std::string> var = initializeData(attr.data());
    code += _tab + var.first + " " + attrName + " = " + var.second + " ;\n";
    code += _tab + opName + ".setAttribute ( attr_" + attrName + ", " +
            attrName + " );\n";
  }

  // Step 3: Add compute function.
  code += _tab + "tensor<" + outType + "> " + nodeName(outs[0]) + " = " +
          opName + ".compute ( " + nodeName(ins[0]) + ", " + nodeName(ins[1]) +
          ");\n";

  return code;
}

std::string dnnc::cppCodeGen::writeTernaryOperator(opNode &computeNode,
                                                   std::vector<node *> &ins,
                                                   std::vector<node *> &outs) {
  std::string code;

  assert(ins.size() == 3 && outs.size() == 1);

  std::string opCode = getOpCodeStr(computeNode.symbol());

  std::string opName = computeNode.name();

  assert(opName.length());

  std::string outType = getDNNC_DataTypeStr(outs[0]->dtype());
  std::string in1Type = getDNNC_DataTypeStr(ins[0]->dtype());
  std::string in2Type = getDNNC_DataTypeStr(ins[1]->dtype());

  // Step 1: Instantiate opterator
  code += _tab + opCode + "<" + outType + ", " + in1Type + ", " + in2Type +
          "> " + opName + "(\"" + opName + "\");\n";

  // Step 2: Add attribute
  for (nodeAttribute attr : computeNode) {
    std::string attrName = getAttrNameStr(attr.name());
    std::pair<std::string, std::string> var = initializeData(attr.data());
    code += _tab + var.first + " " + attrName + " = " + var.second + " ;\n";
    code += _tab + opName + ".setAttribute ( attr_" + attrName + ", " +
            attrName + " );\n";
  }

  // Step 3: Add compute function.
  code += _tab + "tensor<" + outType + "> " + nodeName(outs[0]) + " = " +
          opName + ".compute ( " + nodeName(ins[0]) + ", " + nodeName(ins[1]) +
          ", " + nodeName(ins[2]) + ");\n";

  return code;
}

std::string dnnc::cppCodeGen::writeCustomOperator(opNode &computeNode,
                                                  std::vector<node *> &ins,
                                                  std::vector<node *> &outs) {

  std::string opCode = getOpCodeStr(computeNode.symbol());

  std::string code =
      _tab + "// operator " + opCode + " is not supported yet.\n";
  code += _tab + "// Please file a enhancement request at \n";
  code += _tab +
          "//        https://github.com/ai-techsystems/dnnCompiler/issues \n";
  return code;
}