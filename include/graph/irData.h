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

namespace dnnc {
class irTypeData {
protected:
  IR_DataType _type = IR_DataType::NOTYPE;
  size_t *_ref; /*<! reference count of _data */
  void *_data = 0x0;

public:
  irTypeData(IR_DataType ty, std::vector<int> &d) : _type(ty) {
    assert(ty == IR_DataType::INT8 || ty == IR_DataType::INT16 ||
           ty == IR_DataType::INT32 || ty == IR_DataType::INT64);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<int>(d.begin(), d.end());
  }
  irTypeData(IR_DataType ty, std::vector<float> &d) : _type(ty) {
    assert(ty == IR_DataType::FLOAT || ty == IR_DataType::FLOAT16 ||
           ty == IR_DataType::DOUBLE);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<float>(d.begin(), d.end());
  }
  irTypeData(IR_DataType ty, std::vector<std::string> &d) : _type(ty) {
    assert(ty == IR_DataType::STRING);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<std::string>(d.begin(), d.end());
  }
  irTypeData(IR_DataType ty, std::vector<tensor<bool>> &d) : _type(ty) {
    assert(ty == IR_DataType::BOOL);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<tensor<bool>>(d.begin(), d.end());
  }
  irTypeData(IR_DataType ty, std::vector<tensor<int>> &d)
      : _type(IR_DataType::TENSOR_INT) {
    assert(ty == IR_DataType::INT8 || ty == IR_DataType::INT16 ||
           ty == IR_DataType::INT32 || ty == IR_DataType::INT64);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<tensor<int>>(d.begin(), d.end());
  }
  irTypeData(IR_DataType ty, std::vector<tensor<float>> &d)
      : _type(IR_DataType::TENSOR_FLOAT) {
    assert(ty == IR_DataType::FLOAT || ty == IR_DataType::FLOAT16 ||
           ty == IR_DataType::DOUBLE);
    _ref = new size_t;
    *_ref = 1;
    _data = new std::vector<tensor<float>>(d.begin(), d.end());
  }
  /// \brief copy  constructor
  irTypeData(const irTypeData &other) {
    _ref = other._ref;
    _type = other._type;
    _data = other._data;
    (*_ref)++;
  }
  /// \brief Assignment Operator
  irTypeData &operator=(const irTypeData &other) {
    if (this == &other)
      return *this;

    _ref = other._ref;
    _type = other._type;
    _data = other._data;
    (*_ref)++;

    return *this;
  }
  ~irTypeData() {
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
      case IR_DataType::UINT8:
      case IR_DataType::UINT16:
      case IR_DataType::UINT32:
      case IR_DataType::UINT64:
        delete static_cast<std::vector<unsigned int> *>(_data);
        break;
      case IR_DataType::FLOAT:
      case IR_DataType::FLOAT16:
      case IR_DataType::DOUBLE:
        delete static_cast<std::vector<float> *>(_data);
        break;
      case IR_DataType::STRING:
        delete static_cast<std::vector<std::string> *>(_data);
        break;
      case IR_DataType::TENSOR_BOOL:
        delete static_cast<std::vector<tensor<bool>> *>(_data);
        break;
      case IR_DataType::TENSOR_INT:
        delete static_cast<std::vector<tensor<int>> *>(_data);
        break;
      case IR_DataType::TENSOR_FLOAT:
        delete static_cast<std::vector<tensor<double>> *>(_data);
        break;
      default:
        assert(false && "irTypeData object created without type");
        break;
      }
    }
  }
#ifndef SWIGPYTHON
  operator std::vector<int>() const {
    if (_type != IR_DataType::INT8 && _type != IR_DataType::INT16 &&
        _type != IR_DataType::INT32 && _type != IR_DataType::INT64)
      throw std::bad_cast();

    std::vector<int> ivec = *static_cast<std::vector<int> *>(_data);

    return ivec;
  }
  operator std::vector<unsigned int>() const {
    if (_type != IR_DataType::UINT8 && _type != IR_DataType::UINT16 &&
        _type != IR_DataType::UINT32 && _type != IR_DataType::UINT64)
      throw std::bad_cast();

    std::vector<unsigned int> uivec =
        *static_cast<std::vector<unsigned int> *>(_data);

    return uivec;
  }
  operator std::vector<float>() const {
    if (_type != IR_DataType::FLOAT16 && _type != IR_DataType::FLOAT &&
        _type != IR_DataType::DOUBLE)
      throw std::bad_cast();

    std::vector<float> fvec = *static_cast<std::vector<float> *>(_data);

    return fvec;
  }
  operator std::vector<std::string>() const {
    if (_type != IR_DataType::STRING)
      throw std::bad_cast();

    std::vector<std::string> svec =
        *static_cast<std::vector<std::string> *>(_data);

    return svec;
  }
  operator std::string() const {
    if (_type != IR_DataType::STRING)
      throw std::bad_cast();

    std::vector<std::string> svec =
        *static_cast<std::vector<std::string> *>(_data);

    return svec[0];
  }
  operator std::vector<tensor<bool>>() const {
    if (_type != IR_DataType::TENSOR_BOOL)
      throw std::bad_cast();

    std::vector<tensor<bool>> tbvec =
        *static_cast<std::vector<tensor<bool>> *>(_data);

    if (tbvec.size() == 0)
      throw std::out_of_range("vector of tensor_int with size 0");

    return tbvec;
  }
  operator std::vector<tensor<int>>() const {
    if (_type != IR_DataType::TENSOR_INT)
      throw std::bad_cast();

    std::vector<tensor<int>> tivec =
        *static_cast<std::vector<tensor<int>> *>(_data);

    if (tivec.size() == 0)
      throw std::out_of_range("vector of tensor<int> with size 0");

    return tivec;
  }
  operator std::vector<tensor<double>>() const {
    if (_type != IR_DataType::TENSOR_FLOAT)
      throw std::bad_cast();

    std::vector<tensor<double>> tfvec =
        *static_cast<std::vector<tensor<double>> *>(_data);

    if (tfvec.size() == 0)
      throw std::out_of_range("vector of tensor<float> with size 0");

    return tfvec;
  }
#endif
  IR_DataType type() { return _type; }
};
} // namespace dnnc
