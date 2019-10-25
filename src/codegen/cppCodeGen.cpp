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

#include "codegen/cppCodeGen.h"
#include "graph/inferType.h"
#include <assert.h>
#include <fstream>
#include <regex>

bool dnnc::cppCodeGen::write() {

  inferDataType typeInference(_graph);
  typeInference.main();

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

  std::ofstream out(_bundleDir + FS_PATH_SEPARATOR + _outFile);
  if (!out.is_open() || out.fail()) {
    std::cerr << "ERROR (CODEGEN): could not open file " + _outFile +
                     "to write.\n";
    return false;
  }
  out << writeIncludes() << "\n";
  out << writeMainFunction(code) << "\n";
  out.close();
  return code.length();
}

// \brief replace all characters that can't appear
// in C++ variable name.
std::string dnnc::cppCodeGen::cppName(std::string str) {
  std::string new_str = std::regex_replace(str, std::regex("\\."), "_dot_");
  return new_str;
}

std::string dnnc::cppCodeGen::nodeName(node *n) {
  if (n->ntype() == node::OPERATOR)
    return _prefix + cppName(n->name()) + "_" +
           static_cast<opNode *>(n)->outputs()[0];
  else if (n->ntype() == node::INPUT)
    return _prefix + cppName(n->name());
  else if (n->ntype() == node::OUTPUT)
    return _prefix + cppName(n->name());
  else
    assert(false);
  return _prefix + cppName(n->name());
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

std::string dnnc::cppCodeGen::initializeData(irTypeData dtype,
                                             std::string name) {
  std::string varType;  // int, float, std::vector<float> etc
  std::string initData; // = {1.3, 1.5} etc
  std::string code;     // vector<int> value = {1, 4, 6};
  switch (dtype.type()) {
  case IR_DataType::INT8:
  case IR_DataType::INT16:
  case IR_DataType::INT32:
  case IR_DataType::INT64: {
    varType = getDNNC_IRTypeStr(dtype.type());
    std::vector<int> values = std::vector<int>(dtype);
    if (values.size() == 1) {
      initData = std::to_string(values[0]);
    } else {
      for (auto el : values)
        initData += (initData.size() ? "," : "{") + std::to_string(el);
      initData += values.size() ? "}" : "";
      varType = "std::vector<" + varType + ">";
    }
    code = _tab + varType + " " + name + " = " + initData + " ;\n";
    break;
  }
  case IR_DataType::UINT8:
  case IR_DataType::UINT16:
  case IR_DataType::UINT32:
  case IR_DataType::UINT64: {
    varType = getDNNC_IRTypeStr(dtype.type());
    std::vector<unsigned int> values = std::vector<unsigned int>(dtype);
    if (values.size() == 1) {
      initData = std::to_string(values[0]);
    } else {
      for (auto el : values)
        initData += (initData.size() ? "," : "{") + std::to_string(el);
      initData += values.size() ? "}" : "";
      varType = "std::vector<" + varType + ">";
    }
    code = _tab + varType + " " + name + " = " + initData + " ;\n";
    break;
  }
  case IR_DataType::FLOAT:
  case IR_DataType::FLOAT16:
  case IR_DataType::DOUBLE: {
    varType = getDNNC_IRTypeStr(dtype.type());
    std::vector<float> values = std::vector<float>(dtype);
    if (values.size() == 1) {
      initData = std::to_string(values[0]);
    } else {
      for (auto el : values)
        initData += (initData.size() ? "," : "{") + std::to_string(el);
      initData += values.size() ? "}" : "";
      varType = "std::vector<" + varType + ">";
    }
    code = _tab + varType + " " + name + " = " + initData + " ;\n";
    break;
  }
  case IR_DataType::STRING:
    varType = "std::string";
    initData = std::string(dtype);
    code = _tab + varType + " " + name + " = " + initData + " ;\n";
    break;
  case IR_DataType::TENSOR_BOOL:
    // TODO:
    break;
  case IR_DataType::TENSOR_INT: {
    tensor<long int> values = std::vector<tensor<long int>>(dtype)[0];
    for (auto el : values)
      initData += (initData.size() ? "," : "{") + std::to_string(el);
    initData += values.length() ? "}" : "";
    std::string initVec = name + "_vec";
    initData = "std::vector<long int> " + initVec + " = " + initData + ";\n";
    varType = getDNNC_IRTypeStr(dtype.type());
    code = _tab + initData;
    code += _tab + varType + " " + name + "(1); " + name + ".load(" + initVec +
            ");\n";
    break;
  }
  case IR_DataType::TENSOR_FLOAT: {
    tensor<double> values = std::vector<tensor<double>>(dtype)[0];
    for (auto el : values)
      initData += (initData.size() ? "," : "{") + std::to_string(el);
    initData += values.length() ? "}" : "";
    std::string initVec = name + "_vec";
    initData = "std::vector<double> " + initVec + " = " + initData + ";\n";
    varType = getDNNC_IRTypeStr(dtype.type());
    code = _tab + initData;
    code += _tab + varType + " " + name + "(1); " + name + ".load(" + initVec +
            ");\n";
    break;
  }
  default:
    assert(false && "irTypeData object created without type");
    break;
  }
  return code;
}

std::string dnnc::cppCodeGen::write(dnnParameters param) {
  return initializeData(param.data(), _prefix + cppName(param.name()));
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
  if ((computeNode.symbol() != opConstant &&
       false == computeNode.inputNodes(_graph, ins)) ||
      false == computeNode.outputNodes(_graph, outs)) {
    std::cerr
        << "ERROR (CODEGEN): cound not find all nodes for " << opName << ",\n"
        << "                 an instance of " << opCode << ".\n"
        << "                 Please check model's sanity and try again.\n";
    return code;
  }

  if (ins.size() == 0 && outs.size() == 1) {
    code = writeConstantOperator(computeNode, outs);
  } else if (ins.size() == 1 && outs.size() == 1) {
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

std::string dnnc::cppCodeGen::writeConstantOperator(opNode &computeNode,
                                                    std::vector<node *> &outs) {
  std::string code;

  assert(outs.size() == 1);

  std::string opCode = getOpCodeStr(computeNode.symbol());

  std::string opName = computeNode.name();

  assert(opName.length());

  std::string outType = getDNNC_DataTypeStr(computeNode.dtype());

  // Step 1: Instantiate opterator
  code += "\n";
  code +=
      _tab + opCode + "<" + outType + "> " + opName + "(\"" + opName + "\");\n";

  // Step 2: Add attribute
  for (nodeAttribute attr : computeNode) {
    std::string attrName = getAttrNameStr(attr.name());
    std::string attrVar = opName + "_" + attrName;
    code += initializeData(attr.data(), attrVar);
    code += _tab + opName + ".setAttribute ( attr_" + attrName + ", " +
            attrVar + " );\n";
  }

  // Step 3: Add compute function.
  code += _tab + "tensor<" + outType + "> " + nodeName(&computeNode) + " = " +
          opName + ".compute ();\n";

  return code;
}

std::string dnnc::cppCodeGen::writeUnaryOperator(opNode &computeNode,
                                                 std::vector<node *> &ins,
                                                 std::vector<node *> &outs) {
  std::string code;

  assert(ins.size() == 1 && outs.size() == 1);

  std::string opCode = getOpCodeStr(computeNode.symbol());

  std::string opName = computeNode.name();

  assert(opName.length());

  std::string outType = getDNNC_DataTypeStr(computeNode.dtype());
  std::string inType = getDNNC_DataTypeStr(ins[0]->dtype());

  // Step 1: Instantiate opterator
  code += "\n";
  code += _tab + opCode + "<" + outType + ", " + inType + "> " + opName +
          "(\"" + opName + "\");\n";

  // Step 2: Add attribute
  for (nodeAttribute attr : computeNode) {
    std::string attrName = getAttrNameStr(attr.name());
    std::string attrVar = opName + "_" + attrName;
    code += initializeData(attr.data(), attrVar);
    code += _tab + opName + ".setAttribute ( attr_" + attrName + ", " +
            attrVar + " );\n";
  }

  // Step 3: Add compute function.
  code += _tab + "tensor<" + outType + "> " + nodeName(&computeNode) + " = " +
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

  std::string outType = getDNNC_DataTypeStr(computeNode.dtype());
  std::string in1Type = getDNNC_DataTypeStr(ins[0]->dtype());
  std::string in2Type = getDNNC_DataTypeStr(ins[1]->dtype());

  // Step 1: Instantiate opterator
  code += "\n";
  code += _tab + opCode + "<" + outType + ", " + in1Type + ", " + in2Type +
          "> " + opName + "(\"" + opName + "\");\n";

  // Step 2: Add attribute
  for (nodeAttribute attr : computeNode) {
    std::string attrName = getAttrNameStr(attr.name());
    std::string attrVar = opName + "_" + attrName;
    code += initializeData(attr.data(), attrVar);
    code += _tab + opName + ".setAttribute ( attr_" + attrName + ", " +
            attrVar + " );\n";
  }

  // Step 3: Add compute function.
  code += _tab + "tensor<" + outType + "> " + nodeName(&computeNode) + " = " +
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

  std::string outType = getDNNC_DataTypeStr(computeNode.dtype());
  std::string in1Type = getDNNC_DataTypeStr(ins[0]->dtype());
  std::string in2Type = getDNNC_DataTypeStr(ins[1]->dtype());

  // Step 1: Instantiate opterator
  code += "\n";
  code += _tab + opCode + "<" + outType + ", " + in1Type + ", " + in2Type +
          "> " + opName + "(\"" + opName + "\");\n";

  // Step 2: Add attribute
  for (nodeAttribute attr : computeNode) {
    std::string attrName = getAttrNameStr(attr.name());
    std::string attrVar = opName + "_" + attrName;
    code += initializeData(attr.data(), attrVar);
    code += _tab + opName + ".setAttribute ( attr_" + attrName + ", " +
            attrVar + " );\n";
  }

  // Step 3: Add compute function.
  code += _tab + "tensor<" + outType + "> " + nodeName(&computeNode) + " = " +
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
