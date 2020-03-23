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
#include "core/iterator.h"
#include "core/macros.h"
#include "core/placeHolder.h"

#ifndef SWIGPYTHON
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdlib.h> // malloc, free
#endif
#include <assert.h>
#include <random>

namespace dnnc {
enum INIT_TYPE { INIT_NONE = 0, INIT_RANDOM, INIT_ZERO, INIT_ONE, INIT_VALUE };

template <class To, class Ti1, class Ti2> class baseOperator;
template <typename T> class tensor;
template <typename T> static tensor<T> NULL_TENSOR;

// Tensor with arbitrary rank.
template <typename T> class tensor : public placeHolder<T> {
  template <class To, class Ti1, class Ti2> friend class baseOperator;

protected:
  //////////// protected members /////////////////
  size_t *_ref;   //! reference count of tensor
  T *_mem_layout; //! memory layout of the tensor. TODO: add tiling.

  //////////// protected methods /////////////////

  /// \brief Get the memory for tensor
  T *getMemory(size_t sz) {
    _mem_layout = sz ? static_cast<T *>(malloc(sizeof(T) * sz)) : 0x0;
    if ((sz && !_mem_layout))
      SPDLOG_ERROR("Could not allocate memory for tensor.");
    return _mem_layout;
  }
  /// \brief initialize reference count of the tensor to 1
  void init_ref() {
    _ref = static_cast<size_t *>(malloc(sizeof(size_t)));
    if (!_ref) {
      SPDLOG_ERROR("Could not allocate memory for tensor ref.");
    } else {
      *_ref = 1; // init reference count.
    }
  }

  /// \brief only constructors call init method. Argument type
  /// INIT_TYPE initializes _mem_layout to 0, 1, random or uninitialized.
  void init(INIT_TYPE fill = INIT_NONE, T val = 0) {

    init_ref();

    DIMENSION msize = this->length(); // flat array length
    if (this->rank() == 0 || msize == 0)
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
  tensor(std::vector<DIMENSION> dimn = std::vector<DIMENSION>(),
         std::string n = "", INIT_TYPE fill = INIT_NONE, T init_val = 0)
      : placeHolder<T>(n, dimn), _ref(0x0), _mem_layout(0x0) {
    init(fill, init_val);
  }
  /// USE WITH CAUTION.
  /// CTOR 2: Use this contructor to handover the externally allocated and
  ///         initialized memory to tensor.
  /// This object will own the memory passed to it and free it in the
  /// destructor. This exists solely for performance reasons.
  tensor(T *data, std::vector<DIMENSION> dimn, std::string n = "")
      : placeHolder<T>(n, dimn), _ref(0x0), _mem_layout(data) {
    init_ref();
  }
  /*
  #ifdef SWIGPYTHON
    // CTOR 3: This constructs a tensor of one element and shape(1)
    //         created for convenience for implicit conversion of
    //         numbers to tensor in python interface only.
    tensor(T num, std::string n = "")
        : placeHolder<T>({1}, n), _ref(0x0), _mem_layout(0x0) {
      init(INIT_VALUE, num);
    }
  #endif
  */
  /// \brief Copy Constructor
  tensor(tensor const &other) : placeHolder<T>(other) {
    _ref = other._ref;
    _mem_layout = other._mem_layout;
    (*_ref)++;
  }
  /// \brief Assignment Operator
  tensor &operator=(tensor const &other) {
    // Gracefully handle self assignment
    if (this == &other)
      return *this;

    _ref = other._ref;
    _mem_layout = other._mem_layout;
    (*_ref)++;
    placeHolder<T>::operator=(other);

    return *this;
  }
  /// \brief Comparison Operator
  bool operator==(const tensor &other) {
    if (_mem_layout == other._mem_layout) {
      return this->shape() == other.shape() ? true : false;
    }
    if (this->shape() != other.shape())
      return false;
    for (size_t i = 0; i < this->length(); i++) {
      if (!(_mem_layout[i] == other._mem_layout[i]))
        return false;
    }
    return true;
  }
  ~tensor() {
    if (_ref)
      --(*_ref);
    if (_ref && *_ref == 0 && _mem_layout) {
      free(_ref);
      free(_mem_layout);
    }
  }
  operator bool() const { return this->rank() != 0; }
  /*! Description: creates a deep copy of the tensor
   * Returns: new tensor*/
  tensor<T> copy() const {
    if (isnull())
      return NULL_TENSOR<T>;

    tensor<T> result(this->shape(), this->name());
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

    tensor<newT> result(this->shape(), this->name());

    DIMENSION msize = this->length(); // flat array length
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
                   size_t l = 0, size_t m = 0) {
    this->operator()(i, j, k, l, m) = data;
  }
  /// \brief load 1D vector into the tensor
  void load(std::vector<T> data) {
    size_t sz = this->length();
    for (size_t i = 0; i < data.size() && i < sz; i++)
      _mem_layout[i] = data[i];
  }
  /// \brief UNSAFE METHOD. Load flat array into the tensor.
  ///  UNSAFE because data size MUST be at least as large as tensor length,
  ///  otherwise, it'll lead to crash.
  /// USE WITH CAUTION.
  void load(const T *data) {
    if (!data || isnull())
      return;
    for (size_t i = 0; i < this->length(); i++)
      _mem_layout[i] = data[i];
  }

#ifdef TENSOR_CONVERSION
  /*<! implicit conversion operators are sometimes
   *   needed for compiler. They are a last resort to
   *   complete compilation without error.
   *   They'll eventually be removed, since they cause
   *   performance overhead.
   *   */
  operator tensor<float>() { return asType<float>(); }
  operator tensor<uint8_t>() { return asType<uint8_t>(); }
  operator tensor<int8_t>() { return asType<int8_t>(); }
  operator tensor<uint16_t>() { return asType<uint16_t>(); }
  operator tensor<int16_t>() { return asType<int16_t>(); }
  operator tensor<int32_t>() { return asType<int32_t>(); }
  operator tensor<int64_t>() { return asType<int64_t>(); }
  operator tensor<bool>() { return asType<bool>(); }
  operator tensor<double>() { return asType<double>(); }
  operator tensor<uint32_t>() { return asType<uint32_t>(); }
  operator tensor<uint64_t>() { return asType<uint64_t>(); }
#endif

#ifndef SWIGPYTHON
  /*<! convert tensor to a vector */
  operator std::vector<T>() const {
    std::vector<T> vec;
    for (size_t i = 0; i < this->length(); i++)
      vec.push_back(_mem_layout[i]);
    return vec;
  }

  friend std::ostream &operator<<(std::ostream &os, const tensor<T> &t) {
    if (t.name().size())
      os << t.name() << "=";
    os << t.to_string();
    return os;
  }
  struct it_state {
    size_t pos;
    inline void next(const tensor<T> *ref) { ++pos; }
    inline void begin(const tensor<T> *ref) { pos = 0; }
    inline void end(const tensor<T> *ref) { pos = ref->length(); }
    inline T &get(tensor<T> *ref) { return ref->_mem_layout[pos]; }
    inline const float &get(const tensor<T> *ref) {
      return ref->_mem_layout[pos];
    }
    inline bool cmp(const it_state &s) const { return pos != s.pos; }
  };
  SETUP_ITERATORS(tensor<T>, T &, it_state);
#endif

  std::string to_string(const size_t max_el = DNNC_TENSOR_MAX_EL) const {
    std::string str = this->name().size() ? this->name() + "=" : "";
    if (this->rank() == 0) {
      str += "null tensor";
    } else if (this->rank() == 1 && this->length() == 1) {
      str += std::to_string(_mem_layout[0]);
    } else if (this->rank() == 1) {
      str += "[";
      size_t i = 0;
      for (i = 0; i < this->length() && i < max_el; i++)
        str += (i ? " " : "") + std::to_string(_mem_layout[i]);
      str += i == max_el ? "...]" : "]";
    } else if (this->rank() == 2) {
      str += "[";
      size_t i = 0;
      for (i = 0; i < this->shape()[0] && i < max_el; i++) {
        str += i ? "\n [" : "[";
        size_t j = 0;
        for (j = 0; j < this->shape()[1] && j < max_el; j++) {
          size_t index = i * this->shape()[1] + j;
          str += (j ? " " : "") + std::to_string(_mem_layout[index]);
        }
        str += (j == max_el ? "...]" : "]");
      }
      str += i == max_el ? "...]" : "]";
    } else if (this->rank() == 3) {
      str += "[";
      size_t i = 0;
      for (i = 0; i < this->shape()[0] && i < max_el; i++) {
        str += i ? "\n [" : "[";
        size_t j = 0;
        for (j = 0; j < this->shape()[1] && j < max_el; j++) {
          str += j ? "\n  [" : "[";
          size_t k = 0;
          for (k = 0; k < this->shape()[2] && k < max_el; k++) {
            size_t index = i * this->shape()[1] * this->shape()[2] +
                           j * this->shape()[2] + k;
            str += (k ? " " : "") + std::to_string(_mem_layout[index]);
          }
          str += k == max_el ? "...]" : "]";
        }
        str += j == max_el ? "...]" : "]";
      }
      str += i == max_el ? "...]" : "]";
    } else if (this->rank() == 4) {
      str += "[";
      size_t i = 0;
      for (i = 0; i < this->shape()[0] && i < max_el; i++) {
        str += i ? "\n [" : "[";
        size_t j = 0;
        for (j = 0; j < this->shape()[1] && j < max_el; j++) {
          str += j ? "\n  [" : "[";
          size_t k = 0;
          for (k = 0; k < this->shape()[2] && k < max_el; k++) {
            str += k ? "\n   [" : "[";
            size_t l = 0;
            for (l = 0; l < this->shape()[3] && l < max_el; l++) {
              size_t index =
                  i * this->shape()[1] * this->shape()[2] * this->shape()[3] +
                  j * this->shape()[2] * this->shape()[3] +
                  k * this->shape()[3] + l;
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
                    : std::vector<T>(_mem_layout, _mem_layout + this->length());
  }

  // public methods

  tensor<T> reshape(std::vector<size_t> &new_shape) {
    DIMENSION newLength = new_shape.size() ? 1 : 0;
    for (size_t i = 0; i < new_shape.size(); i++)
      newLength = newLength * new_shape[i];

    // ensure new_shape is same length as original length
    if (newLength == 0)
      SPDLOG_ERROR("new reshape length can't be zero.");
    if (newLength != this->length()) {
      std::string msg = "new reshape length " + std::to_string(newLength) +
                        " does not match tensor\'s original length " +
                        std::to_string(this->length()) + ".\n";
      SPDLOG_ERROR(msg.c_str());
    } else {
      this->_shape = new_shape;
    }
    return *this;
  }
  tensor<T> flatten() {
    if (isnull())
      return NULL_TENSOR<T>;
    std::vector<size_t> new_shape;
    new_shape.push_back(this->length());

    tensor<T> result(new_shape, this->name());
    result.load(_mem_layout);

    return result;
  }
  bool isnull() const { return _mem_layout == 0x0; }
  // TODO:
  void transpose() {}
  // flat index, unsafe method
  T &operator[](const INDEX &index) const {
    if (isnull() || index >= this->length()) {
      std::string msg = "illegal tensor index " + std::to_string(index);
      msg += isnull() ? "on null tensor." : ".";
      throw std::out_of_range(msg.c_str());
      assert(msg.c_str()); // crash and burn.
    }

    return _mem_layout[index];
  }

  T &operator()(std::vector<INDEX> &indices) const {
    INDEX index = 0;
    if (this->rank() == 5 && indices.size() == 5) {
      index =
          indices[0] * this->shape()[1] * this->shape()[2] * this->shape()[3] *
              this->shape()[4] +
          indices[1] * this->shape()[2] * this->shape()[3] * this->shape()[4] +
          indices[2] * this->shape()[3] * this->shape()[4] +
          indices[3] * this->shape()[4] + indices[4];
    } else if (this->rank() == 4 && indices.size() == 4) {
      index =
          indices[0] * this->shape()[1] * this->shape()[2] * this->shape()[3] +
          indices[1] * this->shape()[2] * this->shape()[3] +
          indices[2] * this->shape()[3] + indices[3];
    } else if (this->rank() == 3 && indices.size() == 3) {
      index = indices[0] * this->shape()[1] * this->shape()[2] +
              indices[1] * this->shape()[2] + indices[2];
    } else if (this->rank() == 2 && indices.size() == 2) {
      index = indices[0] * this->shape()[1] + indices[1];
    } else if (this->rank() == 1 && indices.size() == 1) {
      index = indices[0];
    } else {
      if (indices.size() > this->rank()) {
        std::string msg = "number of supplied indices " +
                          std::to_string(indices.size()) +
                          " is more than rank of the tensor " +
                          std::to_string(this->rank()) + ".\n";
        SPDLOG_ERROR(msg.c_str());
      }
      for (size_t i = 0; i < indices.size() && i < this->rank(); i++) {
        DIMENSION dsz = 1;
        for (size_t j = i + 1; j < this->rank(); j++)
          dsz *= this->shape()[j];
        index += indices[i] * dsz;
      }
    }
    return this->operator[](index);
  }

  T &operator()(const INDEX x = 0, const INDEX y = 0, const INDEX z = 0,
                const INDEX w = 0, const INDEX u = 0) const {
    std::vector<INDEX> indices;
    indices.push_back(x);
    if (this->rank() > 1)
      indices.push_back(y);
    if (this->rank() > 2)
      indices.push_back(z);
    if (this->rank() > 3)
      indices.push_back(w);
    if (this->rank() > 4)
      indices.push_back(u);
    return this->operator()(indices);
  }
  bool empty() { return this->length() == 0; }
  std::string dtype() {
    T dummy = 0;
    return dTypeName(dummy);
  }
  std::string to_proto() // return proto string
  {
    std::string tensor_proto = "";
    return tensor_proto;
  }
  T min() const {
    assert(_mem_layout);
    T result = _mem_layout[0];
    for (size_t i = 1; i < this->length(); i++)
      result = result > _mem_layout[i] ? _mem_layout[i] : result;
    return result;
  }
  T max() const {
    assert(_mem_layout);
    T result = _mem_layout[0];
    for (size_t i = 1; i < this->length(); i++)
      result = result < _mem_layout[i] ? _mem_layout[i] : result;
    return result;
  }
  T sum() const {
    T result = 0;
    for (size_t i = 0; i < this->length(); i++)
      result += _mem_layout[i];
    return result;
  }

#ifndef SWIGPYTHON
  // \brief load tensor data from file.
  //        returns true on success, false for failure.
  //        tokens equal to the number of tensor-length
  //        are read, rest are discarded.
  //        if numbers are less, it returns false.
  //        if tokes aren't numbers, it will fail.
  bool read(std::string fileName) {

    std::fstream fs;
    fs.open(fileName, std::ios::in);
    // parameter file could not be opened.
    if (!fs.is_open() || fs.fail()) {
      throw std::runtime_error("Could not open file " + fileName + ".");
      return false;
    }
    std::cout << "reading file " << fileName
              << (this->name().size() ? " for tensor " + this->name() : "")
              << ".\n";
    size_t len = this->length();

    std::string typedStr;
    T fNum;
    size_t index = 0;
    while (std::getline(fs, typedStr)) {
      std::stringstream linestream(typedStr);
      while (linestream >> fNum) {
        if (index >= len) {
          break;
        }
        _mem_layout[index++] = fNum;
      }
      if (index >= len) {
        break;
      }
    }
    fs.close();

    // parameter file did not have parametres equal to tensor length.
    if (index < len) {
      return false;
    }

    return true;
  }
  bool write(std::string fileName) {

    std::fstream fs;
    fs.open(fileName, std::ios::out);
    // parameter file could not be opened.
    if (!fs.is_open() || fs.fail()) {
      throw std::runtime_error("Could not open file " + fileName +
                               " to write.");
      return false;
    }
    std::cout << "writing file " << fileName
              << (this->name().size() ? " for tensor " + this->name() : "")
              << ".\n";
    for (size_t i = 0; i < this->length(); i++) {
      fs << (i ? " " : "") << _mem_layout[i];
    }
    return true;
  }
#endif

}; // class tensor

#ifndef SWIGPYTHON
// \brief return a tensor of type T from a text file with
//        name _name present in bundle directory
//        full of T-type elements.
template <typename T>
tensor<T> readTensor(placeHolder<T> ph, std::string bundleDir = "") {

  std::string fileName =
      bundleDir.empty() ? bundleDir + FS_PATH_SEPARATOR + ph.name() : ph.name();
  std::fstream fs;
  fs.open(fileName, std::ios::in);
  // parameter file could not be opened.
  if (!fs.is_open() || fs.fail()) {
    return NULL_TENSOR<T>;
  }

  std::string typedStr;
  T fNum;
  T *data = new T[ph.length()];
  size_t index = 0;
  while (std::getline(fs, typedStr)) {
    std::stringstream linestream(typedStr);
    while (linestream >> fNum) {
      if (index >= ph.length()) {
        break;
      }
      data[index++] = fNum;
    }
    if (index >= ph.length()) {
      break;
    }
  }
  fs.close();

  // parameter file did not have parametres equal to tensor length.
  if (index < ph.length()) {
    delete[] data;
    return NULL_TENSOR<T>;
  }

  tensor<T> newTensor(ph.shape(), ph.name());
  newTensor.load(data);
  return newTensor;
}

template <typename T> struct tensorCmp {
  bool operator()(const tensor<T> &lhs, const tensor<T> &rhs) {
    return lhs.identifier() < rhs.identifier();
  }
};
#endif

} // namespace dnnc
