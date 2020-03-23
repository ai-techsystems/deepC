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
#include "core/broadcast.h"
#include "operators/macros.h"
#include "operators/opTypes.h"
#include <memory>
#include <type_traits>
#include <vector>

// we're forced to include tensor.h here, because of limitation on
// template instantiations to generate complete definition of the
// operator. This breaks principle of modularity along with my heart. :-/
//
// ONNX operator reference:
//     https://github.com/onnx/onnx/blob/rel-1.5.0/docs/Operators.md
//
#include "core/tensor.h"

namespace dnnc {

template <typename To, typename Ti1, typename Ti2> class baseOperator {
protected:
  OPCODE _op;
  std::string _name;

  template <typename T> T *tensorMem(tensor<T> &t) { return t._mem_layout; }

public:
  baseOperator(OPCODE op, std::string name = "") : _op(op), _name(name) {}

  virtual ~baseOperator() {}

  /*!< return name of the operator */
  virtual inline std::string name() { return _name; }
  /*!< return OPCODE of the operator */
  virtual inline OPCODE symbol() { return _op; }

  // SWIG does not understand, throws error and stops.
  //    Warning 325: Nested class not currently supported (is_one_of ignored)
#ifndef SWIG
  /*!< Constrain data types.*/
  template <typename...> struct is_one_of {
    static constexpr bool value = false;
  };

  template <typename F, typename S, typename... T>
  struct is_one_of<F, S, T...> {
    static constexpr bool value =
        std::is_same<F, S>::value || is_one_of<F, T...>::value;
  };

  /*!<
   \return True if T is one of the types specified else False
   */
  template <typename Kind, typename... Kinds> bool type_check() {
    return is_one_of<Kind, Kinds...>::value;
  }
#endif

  virtual bool getAttribute(OPATTR, float &) { return false; }
  virtual bool getAttribute(OPATTR, int &) { return false; }
  virtual bool getAttribute(OPATTR, std::string &) { return false; }
  virtual bool getAttribute(OPATTR, std::vector<int> &) { return false; }
  virtual bool getAttribute(OPATTR, tensor<int64_t> &obj) { return false; }
  virtual bool getAttribute(OPATTR, tensor<double> &obj) { return false; }
  virtual bool setAttribute(OPATTR, float) { return false; }
  virtual bool setAttribute(OPATTR, int) { return false; }
  virtual bool setAttribute(OPATTR, std::string) { return false; }
  virtual bool setAttribute(OPATTR, std::vector<int>) { return false; }
  virtual bool setAttribute(OPATTR, tensor<int64_t>) { return false; }
  virtual bool setAttribute(OPATTR, tensor<double>) { return false; }

  tensor<To> NOT_SUPPORTED() {
    SPDLOG_ERROR("operator not supported.");
    return NULL_TENSOR<To>;
  }
  virtual tensor<To> compute(tensor<Ti1> in1) { return NOT_SUPPORTED(); }
  virtual tensor<To> compute(tensor<Ti1> &in1) { return NOT_SUPPORTED(); }
  virtual tensor<To> compute(tensor<Ti1> in1, tensor<Ti2> in2) {
    return NOT_SUPPORTED();
  }
  virtual tensor<To> compute(tensor<Ti1> in1, tensor<Ti2> in2,
                             tensor<Ti2> in3) {
    return NOT_SUPPORTED();
  }
};

template <typename To, typename Ti1, typename Ti2> struct opCmp {
  bool operator()(const baseOperator<To, Ti1, Ti2> &lhs,
                  const baseOperator<To, Ti1, Ti2> &rhs) {
    return lhs.symbol() == rhs.symbol() ? lhs.name() < rhs.name()
                                        : lhs.symbol() < rhs.symbol();
  }
};

} // namespace dnnc
