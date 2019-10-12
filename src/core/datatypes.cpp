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

#include "core/datatypes.h"

namespace dnnc {

DNNC_DataType getDNNC_DataType(std::string stype) {
  DNNC_DataType type = NOTYPE;
  if (stype == "float")
    type = FLOAT;
  if (stype == "double")
    type = DOUBLE;
  if (stype == "int32_t")
    type = INT32;
  if (stype == "uint8_t")
    type = UINT8;
  if (stype == "int16_t")
    type = INT16;
  if (stype == "int8_t")
    type = INT8;
  if (stype == "string")
    type = STRING;
  if (stype == "int64_t")
    type = INT64;
  if (stype == "bool")
    type = BOOL;
  if (stype == "uint16_t")
    type = UINT16;
  if (stype == "half")
    type = FLOAT16;
  if (stype == "uint32_t")
    type = UINT32;
  if (stype == "int64_t")
    type = UINT64;
  return type;
}

std::string getDNNC_DataTypeStr(DNNC_DataType dtype) {
  std::string type = "";
  switch (dtype) {
  case FLOAT:
    type = "float";
    break;
  case DOUBLE:
    type = "double";
    break;
  case INT32:
    type = "int32_t";
    break;
  case UINT8:
    type = "uint8_t";
    break;
  case INT16:
    type = "int16_t";
    break;
  case INT8:
    type = "int8_t";
    break;
  case STRING:
    type = "string";
    break;
  case INT64:
    type = "int64_t";
    break;
  case BOOL:
    type = "bool";
    break;
  case FLOAT16:
  case BFLOAT16:
    type = "half";
    break;
  case UINT16:
    type = "uint16_t";
    break;
  case UINT32:
    type = "uint32_t";
    break;
  case UINT64:
    type = "int64_t";
    break;
  case NOTYPE:
  default:
    type = "notype";
    break;
  }
  return type;
}

IR_DataType getDNNC_IRType(std::string stype) {
  IR_DataType type = IR_DataType::NOTYPE;
  if (stype == "float")
    type = IR_DataType::FLOAT;
  if (stype == "double")
    type = IR_DataType::DOUBLE;
  if (stype == "int32_t")
    type = IR_DataType::INT32;
  if (stype == "uint8_t")
    type = IR_DataType::UINT8;
  if (stype == "int16_t")
    type = IR_DataType::INT16;
  if (stype == "int8_t")
    type = IR_DataType::INT8;
  if (stype == "string")
    type = IR_DataType::STRING;
  if (stype == "int64_t")
    type = IR_DataType::INT64;
  if (stype == "bool")
    type = IR_DataType::BOOL;
  if (stype == "uint16_t")
    type = IR_DataType::UINT16;
  if (stype == "half")
    type = IR_DataType::FLOAT16;
  if (stype == "uint32_t")
    type = IR_DataType::UINT32;
  if (stype == "int64_t")
    type = IR_DataType::UINT64;
  if (stype == "tensor<bool>")
    type = IR_DataType::TENSOR_BOOL;
  if (stype == "tensor<int>")
    type = IR_DataType::TENSOR_INT;
  if (stype == "tensor<float>")
    type = IR_DataType::TENSOR_FLOAT;
  if (stype == "graph")
    type = IR_DataType::GRAPH;
  return type;
}

std::string getDNNC_IRTypeStr(IR_DataType dtype) {
  std::string type = "";
  switch (dtype) {
  case IR_DataType::FLOAT:
    type = "float";
    break;
  case IR_DataType::DOUBLE:
    type = "double";
    break;
  case IR_DataType::INT32:
    type = "int32_t";
    break;
  case IR_DataType::UINT8:
    type = "uint8_t";
    break;
  case IR_DataType::INT16:
    type = "int16_t";
    break;
  case IR_DataType::INT8:
    type = "int8_t";
    break;
  case IR_DataType::STRING:
    type = "string";
    break;
  case IR_DataType::INT64:
    type = "int64_t";
    break;
  case IR_DataType::BOOL:
    type = "bool";
    break;
  case IR_DataType::FLOAT16:
  case IR_DataType::BFLOAT16:
    type = "half";
    break;
  case IR_DataType::UINT16:
    type = "uint16_t";
    break;
  case IR_DataType::UINT32:
    type = "uint32_t";
    break;
  case IR_DataType::UINT64:
    type = "int64_t";
    break;
  case IR_DataType::TENSOR_BOOL:
    type = "tensor<bool>";
    break;
  case IR_DataType::TENSOR_INT:
    type = "tensor<int>";
    break;
  case IR_DataType::TENSOR_FLOAT:
    type = "tensor<float>";
    break;
  case IR_DataType::GRAPH:
    type = "graph";
    break;
  case IR_DataType::NOTYPE:
  default:
    type = "notype";
    break;
  }
  return type;
}
} // namespace dnnc
