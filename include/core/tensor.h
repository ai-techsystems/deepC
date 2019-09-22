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
#include <assert.h>
#include <random>

namespace dnnc {
typedef size_t INDEX;
typedef size_t DIMENSION;
enum INIT_TYPE { INIT_NONE = 0, INIT_RANDOM, INIT_ZERO, INIT_ONE, INIT_VALUE };

template <class To, class Ti1, class Ti2> class baseOperator;
template <typename T> class tensor;
template <typename T> static tensor<T> NULL_TENSOR;

// Tensor with arbitrary rank.
template <typename T> class tensor {
  template <class To, class Ti1, class Ti2> friend class baseOperator;

protected:
  //////////// protected members /////////////////
  size_t *_ref;      //! reference count of tensor
  T *_mem_layout;    //! memory layout of the tensor. TODO: add tiling.
  std::string _name; //! tensor name
  std::vector<DIMENSION> _shape; //! tensor shape

  //////////// protected methods /////////////////

  /// \brief Get the memory for tensor
  T *getMemory(size_t sz) {
    _mem_layout = sz ? static_cast<T *>(malloc(sizeof(T) * sz)) : 0x0;
    if ((sz && !_mem_layout))
      throw std::bad_alloc();
    return _mem_layout;
  }
  /// \brief initialize reference count of the tensor to 1
  void init_ref() {
    _ref = static_cast<size_t *>(malloc(sizeof(size_t)));
    if (!_ref)
      throw std::bad_alloc();
    *_ref = 1; // init reference count.
  }

  /// \brief only constructors call init method. Argument type
  /// INIT_TYPE initializes _mem_layout to 0, 1, random or uninitialized.
  void init(INIT_TYPE fill = INIT_NONE, T val = 0) {

    init_ref();

    DIMENSION msize = length(); // flat array length
    if (rank() == 0 || msize == 0)
      return;

    _mem_layout = getMemory(msize);

    // initilize with normal distribution.
    if (fill == INIT_NONE) {
      ; // no fill
    } else if (fill == INIT_RANDOM) {
      std::default_random_engine generator;
      std::normal_distribution<double> distribution(127.5, 20.0);
      for (size_t i = 0; i < msize; i++)
        _mem_layout[i] = static_cast<T>(distribution(generator));
    } else if (fill == INIT_ZERO) {
      for (size_t i = 0; i < msize; i++)
        _mem_layout[i] = static_cast<T>(0);
    } else if (fill == INIT_ONE) {
      for (size_t i = 0; i < msize; i++)
        _mem_layout[i] = static_cast<T>(1);
    } else if (fill == INIT_VALUE) {
      for (size_t i = 0; i < msize; i++)
        _mem_layout[i] = val;
    }
  }

public:
  /// tensor constructor with arbitrary dimension up to 4.

  /// CTOR 1: Use this contructor with shape vector and to initialize
  ///         with zero, one, or random numbers.
  tensor(std::vector<DIMENSION> dimn, std::string n = "",
         INIT_TYPE fill = INIT_NONE, T init_val = 0)
      : _ref(0x0), _mem_layout(0x0), _name(n), _shape(dimn) {
    init(fill, init_val);
  }
  /// CTOR 1a: Use this contructor with upto 4 dimensions to initialize with
  ///          zero, one, or random numbers.
  tensor(DIMENSION x = 0, DIMENSION y = 0, DIMENSION z = 0, DIMENSION w = 0,
         std::string n = "", INIT_TYPE fill = INIT_NONE, T init_val = 0)
      : _ref(0x0), _mem_layout(0x0), _name(n) {
    if (x) {
      _shape.push_back(x);
      if (y)
        _shape.push_back(y);
      if (z)
        _shape.push_back(z);
      if (w)
        _shape.push_back(w);
    }
    init(fill, init_val);
  }
  /// USE WITH CAUTION.
  /// CTOR 2: Use this contructor to handover the externally allocated and
  ///         initialized memory to tensor.
  /// This object will own the memory passed to it and free it in the
  /// destructor. This exists solely for performance reasons.
  tensor(T *data, std::vector<DIMENSION> dimn, std::string n = "")
      : _ref(0x0), _mem_layout(data), _name(n), _shape(dimn) {
    init_ref();
  }
  /// \brief Copy Constructor
  tensor(const tensor &other) {
    _ref = other._ref;
    _name = other._name;
    _shape = other._shape;
    _mem_layout = other._mem_layout;
    (*_ref)++;
  }
  /// \brief Assignment Operator
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
  /// \brief Comparison Operator
  /* bool operator==(const tensor &other) {
    if (_mem_layout == other._mem_layout) {
      return _shape == other._shape ? true : false;
    }
    if (_shape != other._shape)
      return false;
    for (size_t i = 0; i < length(); i++) {
      if (!(_mem_layout[i] == other._mem_layout[i]))
        return false;
    }
    return true;
  } */
  ~tensor() {
    if (_ref)
      --(*_ref);
    if (_ref && *_ref == 0 && _mem_layout) {
      free(_ref);
      free(_mem_layout);
    }
  }
  operator bool() const { return rank() != 0; }
  /*! Description: creates a deep copy of the tensor
   * Returns: new tensor*/
  tensor<T> copy() const {
    if (isnull())
      return NULL_TENSOR<T>;

    tensor<T> result(_shape, _name);
    result.load(_mem_layout);

    return result;
  }
  /// \brief invert the sign of each element of the tensor
  tensor<T> negate() const {
    tensor<T> result = copy();
    DIMENSION msize = result.length(); // flat array length
    for (size_t i = 0; i < msize; i++)
      result._mem_layout[i] = -_mem_layout[i];

    return result;
  }
  /// \brief absolute value of each element of the tensor
  tensor<T> absolute() const {
    tensor<T> result = copy();
    DIMENSION msize = result.length(); // flat array length
    for (size_t i = 0; i < msize; i++)
      result._mem_layout[i] =
          _mem_layout[i] < static_cast<T>(0) ? -_mem_layout[i] : _mem_layout[i];

    return result;
  }
  /// \brief identifier of the tensor
  size_t identifier() const {
    return reinterpret_cast<size_t>(_mem_layout - 0xfff);
  }
  /// \brief check if this tensor has same id as other.
  bool sameas(const tensor<T> &other) const {
    return identifier() == other.identifier();
  }
  /// \brief Return copy of the tensor, cast to a specified type.
  template <typename newT> tensor<newT> asType() {
    // if (typeid(T) == typeid(newT))
    //  return *dynamic_cast<tensor<newT*>>(this);

    tensor<newT> result(_shape, _name);

    DIMENSION msize = length(); // flat array length
    for (size_t i = 0; i < msize; i++)
      result[i] = _mem_layout[i];
    return result;
  }
  /// \brief return a copy of the tensor, cast to double
  tensor<double> asTypeDouble() { return asType<double>(); }
  /// \brief return a copy of the tensor, cast to float
  tensor<float> asTypeFloat() { return asType<float>(); }
  /// \brief return a copy of the tensor, cast to int
  tensor<int> asTypeInt() { return asType<int>(); }
  /// \brief return a copy of the tensor, cast to uint8
  tensor<uint8_t> asTypeUint8() { return asType<uint8_t>(); }
  /// \brief return a copy of the tensor, cast to long
  tensor<long> asTypeLong() { return asType<long>(); }
  /// \brief return a copy of the tensor, cast to unsigned long
  tensor<size_t> asTypeULong() { return asType<size_t>(); }
  /// \brief return a copy of the tensor, cast to bool
  tensor<bool> asTypeBool() { return asType<bool>(); }

  /// \brief load single data into tensor.
  inline void load(const T &data, std::vector<size_t> indices) {
    this->operator()(indices) = data;
  }
  inline void load(const T &data, size_t i, size_t j = 0, size_t k = 0,
                   size_t l = 0) {
    this->operator()(i, j, k, l) = data;
  }
  /// \brief load 1D vector into the tensor
  void load(std::vector<T> data) {
    size_t sz = length();
    for (size_t i = 0; i < data.size() && i < sz; i++)
      _mem_layout[i] = data[i];
  }
  /// \brief UNSAFE METHOD. Load flat array into the tensor.
  ///  UNSAFE because data size MUST be at least as large as tensor length,
  ///  otherwise, it'll lead to crash.
  /// USE WITH CAUTION.
  void load(T *data) {
    if (!data || isnull())
      return;
    for (size_t i = 0; i < length(); i++)
      _mem_layout[i] = data[i];
  }
#ifndef SWIGPYTHON
  friend std::ostream &operator<<(std::ostream &os, const tensor<T> &t) {
    if (t._name.size())
      os << t._name << "=";
    os << t.to_string();
    return os;
  }
#endif

  std::string to_string(const size_t max_el = DNNC_TENSOR_MAX_EL) const {
    std::string str = _name.size() ? _name + "=" : "";
    if (rank() == 0) {
      str += "null tensor";
    } else if (rank() == 1) {
      str += "[";
      size_t i = 0;
      for (i = 0; i < length() && i < max_el; i++)
        str += (i ? " " : "") + std::to_string(_mem_layout[i]);
      str += i == max_el ? "...]" : "]";
    } else if (rank() == 2) {
      str += "[";
      size_t i = 0;
      for (i = 0; i < _shape[0] && i < max_el; i++) {
        str += i ? "\n [" : "[";
        size_t j = 0;
        for (j = 0; j < _shape[1] && j < max_el; j++) {
          size_t index = i * _shape[1] + j;
          str += (j ? " " : "") + std::to_string(_mem_layout[index]);
        }
        str += (j == max_el ? "...]" : "]");
      }
      str += i == max_el ? "...]" : "]";
    } else if (rank() == 3) {
      str += "[";
      size_t i = 0;
      for (i = 0; i < _shape[0] && i < max_el; i++) {
        str += i ? "\n [" : "[";
        size_t j = 0;
        for (j = 0; j < _shape[1] && j < max_el; j++) {
          str += j ? "\n  [" : "[";
          size_t k = 0;
          for (k = 0; k < _shape[2] && k < max_el; k++) {
            size_t index = i * _shape[1] * _shape[2] + j * _shape[2] + k;
            str += (k ? " " : "") + std::to_string(_mem_layout[index]);
          }
          str += k == max_el ? "...]" : "]";
        }
        str += j == max_el ? "...]" : "]";
      }
      str += i == max_el ? "...]" : "]";
    } else if (rank() == 4) {
      str += "[";
      size_t i = 0;
      for (i = 0; i < _shape[0] && i < max_el; i++) {
        str += i ? "\n [" : "[";
        size_t j = 0;
        for (j = 0; j < _shape[1] && j < max_el; j++) {
          str += j ? "\n  [" : "[";
          size_t k = 0;
          for (k = 0; k < _shape[2] && k < max_el; k++) {
            str += k ? "\n   [" : "[";
            size_t l = 0;
            for (l = 0; l < _shape[3] && l < max_el; l++) {
              size_t index = i * _shape[1] * _shape[2] * _shape[3] +
                             j * _shape[2] * _shape[3] + k * _shape[3] + l;
              str += (l ? " " : "") + std::to_string(_mem_layout[index]);
            }
            str += l == max_el ? "...]" : "]";
          }
          str += k == max_el ? "...]" : "]";
        }
        str += j == max_el ? "...]" : "]";
      }
      str += i == max_el ? "...]" : "]";
    }
    return str;
  }

  /// \brief return 1D flat array
  const std::vector<T> data() const {
    return isnull() ? std::vector<T>()
                    : std::vector<T>(_mem_layout, _mem_layout + length());
  }

  // public methods

  /// \brief Return number of elements in the tensor.
  const DIMENSION length() const {
    DIMENSION sz = rank() ? 1 : 0;
    for (size_t i = 0; i < rank(); i++)
      sz = sz * _shape[i];
    return sz;
  }
  std::string name() const { return _name; }
  const DIMENSION rank() const { return _shape.size(); }
  const std::vector<DIMENSION> shape() const { return _shape; }
  tensor<T> reshape(std::vector<size_t> &new_shape) {
    DIMENSION newLength = new_shape.size() ? 1 : 0;
    for (size_t i = 0; i < new_shape.size(); i++)
      newLength = newLength * new_shape[i];

    // ensure new_shape is same length as original length
    if (newLength == 0)
      throw std::invalid_argument("new reshape length can't be zero.");
    if (newLength != length()) {
      std::string msg = "new reshape length " + std::to_string(newLength) +
                        " does not match tensor\'s original length " +
                        std::to_string(length()) + ".\n";
      throw std::invalid_argument(msg);
    } else {
      _shape = new_shape;
    }
    return *this;
  }
  /*! It flattens tensor in place, reducing the tensor's rank to
   * 1 as in flat 1D array */
  void flatteninplace() {
    DIMENSION sz = length();
    _shape.clear();
    _shape.push_back(sz);
    return;
  }
  tensor<T> flatten() {
    if (isnull())
      return NULL_TENSOR<T>;
    std::vector<size_t> new_shape;
    new_shape.push_back(length());

    tensor<T> result(new_shape, _name);
    result.load(_mem_layout);

    return result;
  }
  bool isnull() const { return _mem_layout == 0x0; }
  // TODO:
  void transpose() {}
  // flat index, unsafe method
  T &operator[](const INDEX &index) const {
    if (isnull() || index >= length()) {
      std::string msg = "illegal tensor index " + std::to_string(index);
      msg += isnull() ? "on null tensor." : ".";
      throw std::out_of_range(msg.c_str());
      assert(msg.c_str()); // crash and burn.
    }

    return _mem_layout[index];
  }

  T &operator()(std::vector<INDEX> &indices) const {
    INDEX index = 0;
    if (rank() == 4 && indices.size() == 4) {
      index = indices[0] * _shape[1] * _shape[2] * _shape[3] +
              indices[1] * _shape[2] * _shape[3] + indices[2] * _shape[3] +
              indices[3];
    } else if (rank() == 3 && indices.size() == 3) {
      index = indices[0] * _shape[1] * _shape[2] + indices[1] * _shape[2] +
              indices[2];
    } else if (rank() == 2 && indices.size() == 2) {
      index = indices[0] * _shape[1] + indices[1];
    } else if (rank() == 1 && indices.size() == 1) {
      index = indices[0];
    } else {
      if (indices.size() > rank()) {
        std::string msg = "number of supplied indices " +
                          std::to_string(indices.size()) +
                          " is more than rank of the tensor " +
                          std::to_string(rank()) + ".\n";
        throw std::invalid_argument(msg);
      }
      for (size_t i = 0; i < indices.size() && i < rank(); i++) {
        DIMENSION dsz = 1;
        for (size_t j = i + 1; j < rank(); j++)
          dsz *= _shape[j];
        index += indices[i] * dsz;
      }
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

}; // class tensor

template <typename T> struct tensorCmp {
  bool operator()(const tensor<T> &lhs, const tensor<T> &rhs) {
    return lhs.identifier() < rhs.identifier();
  }
};

} // namespace dnnc
