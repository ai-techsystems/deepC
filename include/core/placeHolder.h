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

#ifndef SWIGPYTHON
#include <string>
#include <vector>
#endif

namespace dnnc {

template <typename T> class placeHolder {

protected:
  std::string _name;             //! placeHolder name
  std::vector<DIMENSION> _shape; //! placeHolder shape

  void init(std::vector<DIMENSION> &dimn) {
    for (auto num : dimn) {
      if (num == 0)
        break;
      _shape.push_back(num);
    }
  }
  // class tensor uses this constructor to allow for
  // _shape with no dimension, i.e. rank 0.
  placeHolder(std::string n, std::vector<DIMENSION> dimn) : _name(n) {
    init(dimn);
  }

public:
  placeHolder(std::vector<DIMENSION> dimn, std::string n = "") : _name(n) {
    init(dimn);
  }

  void name(std::string n) { _name = n; }
  std::string name() const { return _name; }
  const DIMENSION rank() const { return _shape.size(); }
  const std::vector<DIMENSION> shape() const { return _shape; }
  /// \brief Return number of elements in the tensor.
  const DIMENSION length() const {
    DIMENSION sz = rank() ? 1 : 0;
    for (size_t i = 0; i < rank(); i++)
      sz = sz * _shape[i];
    return sz;
  }
  /*<! It flattens tensor in place, reducing the tensor's rank to
   * 1 as in flat 1D array */
  void flatteninplace() {
    DIMENSION sz = length();
    _shape.clear();
    _shape.push_back(sz);
    return;
  }
};
} // namespace dnnc
