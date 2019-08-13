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

#include "core/datatypes.h"
#include "core/macros.h"

#ifndef SWIGPYTHON
#include <iostream>
#include <stdexcept>
#include <stdlib.h> // malloc, free
#include <string>
#include <vector>
#endif
#include <random>

namespace dnnc {
typedef size_t INDEX;
typedef size_t DIMENSION;

template <typename T> class baseOperator;

// Tensor with arbitrary rank.
template <typename T> class tensor {
  friend class baseOperator<T>;

protected:
  //////////// protected members /////////////////
  size_t *_ref;
  std::string _name;
  std::vector<DIMENSION> _shape;
  T *_mem_layout; // TODO: add tiling.

  //////////// protected methods /////////////////
  T *getMemory(size_t sz) {
    _mem_layout = sz ? static_cast<T *>(malloc(sizeof(T) * sz)) : 0x0;
    _ref = static_cast<size_t *>(malloc(sizeof(size_t)));
    if ((sz && !_mem_layout) || !_ref)
      throw std::bad_alloc();
    return _mem_layout;
  }
  // only constructors  call init method
  void init() {
    size_t msize = length(); // flat array length
    if (rank() == 0)
#ifndef SWIGPYTHON
      throw std::invalid_argument("tensor with no shape.");
#endif
    _mem_layout = getMemory(msize);

    *_ref = 1; // init reference count.
    // initilize with normal distribution.
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(127.5, 20.0);
    for (size_t i = 0; i < msize; i++)
      _mem_layout[i] = static_cast<T>(distribution(generator));
  }

public:
  // tensor constructor with arbitrary dimension
  tensor(std::vector<DIMENSION> dimn, std::string n = "")
      : _name(n), _mem_layout(0x0) {
    _shape = dimn;
    init();
  }
  tensor(DIMENSION x = 0, DIMENSION y = 0, DIMENSION z = 0, DIMENSION w = 0,
         std::string n = "")
      : _name(n), _mem_layout(0x0) {
    if (x) {
      _shape.push_back(x);
      if (y)
        _shape.push_back(y);
      if (z)
        _shape.push_back(z);
      if (w)
        _shape.push_back(w);
    }
    init();
  }
  tensor(const tensor &other) {
    _ref = other._ref;
    _name = other._name;
    _shape = other._shape;
    _mem_layout = other._mem_layout;
    (*_ref)++;
  }
  tensor &operator=(const tensor &other) {
    // Gracefully handle self assignment
    if (this == &other)
      return *this;

    _ref = other._ref;
    _name = other._name;
    _shape = other._shape;
    _mem_layout = other._mem_layout;
    (*_ref)++;

    return *this;
  }
  ~tensor() {
    --(*_ref);
    if (*_ref == 0 && _mem_layout) {
      free(_ref);
      free(_mem_layout);
    }
  }

  // WARNING: Make sure data being loaded has same size as tensor.
  void load(T *data) {
    if (!data)
      return;
    for (size_t i = 0; i < length(); i++)
      _mem_layout[i] = data[i];
  }
  friend std::ostream &operator<<(std::ostream &os, const tensor<T> &t) {
    if (t._name.size())
      os << t._name << "=";
    for (size_t i = 0; i < t.length(); i++)
      os << t._mem_layout[i] << ' ';
    return os;
  }
  std::string to_string() {
    std::string str;
    if (_name.size())
      str += _name + "=";
    for (size_t i = 0; i < length(); i++)
      str += std::to_string(_mem_layout[i]) + (i == length() - 1 ? "" : " ");
    return str;
  }

  // public methods

  const DIMENSION length() const {
    DIMENSION sz = rank() ? 1 : 0;
    for (size_t i = 0; i < rank(); i++)
      sz = sz * _shape[i];
    return sz;
  }
  std::string name() const { return _name; }
  const DIMENSION rank() const { return _shape.size(); }
  const std::vector<DIMENSION> shape() const { return _shape; }
  void reshape(std::vector<size_t> &new_shape) {
    DIMENSION newLength = new_shape.size() ? 1 : 0;
    for (size_t i = 0; i < new_shape.size(); i++)
      newLength = newLength * new_shape[i];

    // ensure new_shape is same length as original length
    if (newLength == 0)
      throw std::invalid_argument("new reshape length can't be zero.");
    if (newLength != length()) {
      std::string msg = "new reshape length " + std::to_string(newLength) +
                        " does not match tensor\'s length" +
                        std::to_string(length()) + ".\n";
      throw std::invalid_argument(msg);
    }

    _shape = new_shape;
  }
  void broadcast(unsigned int n) {
    // INCREASE the rank by duplicating last axis.
  }
  // flat index, unsafe method
  T &operator[](const INDEX &index) const {
    if (index >= length())
      throw std::out_of_range("illegal tensor index.");

    return _mem_layout[index];
  }

  T &operator()(std::vector<INDEX> &indices) const {
    INDEX index = 0;
    for (size_t i = 0; i < indices.size(); i++) {
      DIMENSION dsz = 1;
      for (size_t j = i + 1; j < rank(); j++)
        dsz *= _shape[j];
      index += indices[i] * dsz;
    }
    return this->operator[](index);
  }
  T &operator()(const INDEX x = 0, const INDEX y = 0, const INDEX z = 0,
                const INDEX w = 0) const {
    std::vector<INDEX> indices;
    indices.push_back(x);
    if (rank() > 1)
      indices.push_back(y);
    if (rank() > 2)
      indices.push_back(z);
    if (rank() > 3)
      indices.push_back(w);

    return this->operator()(indices);
  }
  std::string dtype() { return dtype_str[typeid(T).name()[0] - 'a']; }
  bool empty() { return length() == 0; }
  std::string to_proto() // return proto string
  {
    std::string tensor_proto = "";
    return tensor_proto;
  }
};
} // namespace dnnc
