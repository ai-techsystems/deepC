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
#pragma once

#include "core/tensor.h"
#include "operators/baseOperator.h"
#include <any>
#include <vector>

namespace dnnc {

class genericData {
protected:
  IR_DataType _type = IR_DataType::NOTYPE;
  size_t *_ref; /*<! reference count of _data */
  void *_data = 0x0;

public:
  genericData(IR_DataType ty, std::vector<int> &d) : _type(ty) {
    assert(ty == IR_DataType::INT8 || ty == IR_DataType::INT16 ||
           ty == IR_DataType::INT32 || ty == IR_DataType::INT64);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<int>(d.begin(), d.end());
  }
  genericData(IR_DataType ty, std::vector<float> &d) : _type(ty) {
    assert(ty == IR_DataType::FLOAT || ty == IR_DataType::FLOAT16 ||
           ty == IR_DataType::DOUBLE);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<float>(d.begin(), d.end());
  }
  genericData(IR_DataType ty, std::vector<std::string> &d) : _type(ty) {
    assert(ty == IR_DataType::STRING);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<std::string>(d.begin(), d.end());
  }
  genericData(IR_DataType ty, std::vector<tensor<bool>> &d) : _type(ty) {
    assert(ty == IR_DataType::BOOL);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<tensor<bool>>(d.begin(), d.end());
  }
  genericData(IR_DataType ty, std::vector<tensor<int>> &d)
      : _type(IR_DataType::TENSOR_INT) {
    assert(ty == IR_DataType::INT8 || ty == IR_DataType::INT16 ||
           ty == IR_DataType::INT32 || ty == IR_DataType::INT64);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<tensor<int>>(d.begin(), d.end());
  }
  genericData(IR_DataType ty, std::vector<tensor<float>> &d)
      : _type(IR_DataType::TENSOR_FLOAT) {
    assert(ty == IR_DataType::FLOAT || ty == IR_DataType::FLOAT16 ||
           ty == IR_DataType::DOUBLE);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<tensor<float>>(d.begin(), d.end());
  }
  /// \brief copy  constructor
  genericData(const genericData &other) {
    _ref = other._ref;
    _type = other._type;
    _data = other._data;
    (*_ref)++;
  }
  /// \brief Assignment Operator
  genericData &operator=(const genericData &other) {
    if (this == &other)
      return *this;

    _ref = other._ref;
    _type = other._type;
    _data = other._data;
    (*_ref)++;

    return *this;
  }
  ~genericData() {
    if (_ref)
      --(*_ref);
    if (_ref && *_ref == 0 && _data) {
      free(_ref);
      switch (_type) {
      case IR_DataType::INT8:
      case IR_DataType::INT16:
      case IR_DataType::INT32:
      case IR_DataType::INT64:
        delete static_cast<std::vector<int> *>(_data);
        break;
      case IR_DataType::FLOAT:
      case IR_DataType::FLOAT16:
      case IR_DataType::DOUBLE:
        delete static_cast<std::vector<float> *>(_data);
        break;
      case IR_DataType::STRING:
        delete static_cast<std::string *>(_data);
        break;
      case IR_DataType::TENSOR_BOOL:
        delete static_cast<std::vector<tensor<bool>> *>(_data);
        break;
      case IR_DataType::TENSOR_INT:
        delete static_cast<std::vector<tensor<int>> *>(_data);
        break;
      case IR_DataType::TENSOR_FLOAT:
        delete static_cast<std::vector<tensor<float>> *>(_data);
        break;
      default:
        assert(false && "genericData object created without type");
        break;
      }
    }
  }
  operator int() const {
    if (_type != IR_DataType::INT64)
      throw std::bad_cast();

    std::vector<int> ivec = *static_cast<std::vector<int> *>(_data);

    if (ivec.size() == 0)
      throw std::out_of_range("vector of size 0");

    return ivec[0];
  }
};

class nodeAttribute {
protected:
  OPATTR _name = attr_invalid;
  genericData _value;

public:
  nodeAttribute(OPATTR n, genericData &v) : _name(n), _value(v) {}
};

/*! Graph node
 * */
/*! Compute Graph Node.
 *       It represents basic computational unit (like adder/multiplier)
 * available in underlying hardware.
 * */
class node {
protected:
  // TODO: add node attributes like level, graph-input, graph-output,
  //       placeholder, const, variable etc.
  OPCODE _symbol;    /*!< operator aka symbol */
  std::string _name; /*! node name */
  std::vector<std::string>
      _input_names; /*!< inputs, i.e. tensors coming to   this node */
  std::vector<std::string>
      _output_names; /*!< outputs, i.e tensors going  from this node */
  std::vector<nodeAttribute> _attributes; /*!< attributes of the node, i.e.
                                        values that don't flow in and out */

  node() = delete; /*!< default constructor not allowed */
public:
  node(OPCODE sym, std::string n = "") : _symbol(sym), _name(n) {}
  ~node() {}

  void addInput(std::string n) { _input_names.push_back(n); }
  void addOutput(std::string n) { _output_names.push_back(n); }
  void addAttribute(nodeAttribute &attr) { _attributes.push_back(attr); }
};

} // namespace dnnc
