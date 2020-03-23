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

#pragma once

#include <string>
#include <typeinfo>

namespace dnnc {
// enum for target machine.
#define DNNC_Basic_DType                                                       \
  /*<! This code is for ONNX TensorProto.DataType                              \
       Reference: onnx/onnx.proto3, line 319 */                                \
  NOTYPE = 0, /*!< invalid */                                                  \
      FLOAT,  /*!< float */                                                    \
      UINT8,  /*!< uint8_t */                                                  \
      INT8,   /*!< int8_t */                                                   \
      UINT16, /*!< uint16_t */                                                 \
      INT16,  /*!< int16_t */                                                  \
      INT32,  /*!< int32_t */                                                  \
      INT64,  /*!< int64_t */                                                  \
      STRING, /*!< string */                                                   \
      BOOL,   /*!< bool */                                                     \
                                                                               \
      /*!< IEEE754 half-precision floating-point format (16 bits wide).        \
           This format has 1 sign bit, 5 exponent bits, and 10 mantissa        \
         bits.*/                                                               \
      FLOAT16,    /*! half-float */                                            \
      DOUBLE,     /*! double precision, aka float64 */                         \
      UINT32,     /*! uint32_t */                                              \
      UINT64,     /*! uint64_t */                                              \
      COMPLEX64,  /*!< complex with float32 real and imaginary components */   \
      COMPLEX128, /*!< complex with float64 real and imaginary components */   \
                                                                               \
      /*!< Non-IEEE floating-point format based on IEEE754 single-precision    \
           floating-point number truncated to 16 bits.                         \
           This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.   \
       */                                                                      \
      BFLOAT16

enum DNNC_DataType { DNNC_Basic_DType };

/*!< reserved for advanced usage for nodes that represent multiple/generic
 * types. Don't use it, if you don't know what it means.
 * */
enum class IR_DataType {
  DNNC_Basic_DType,
  TENSOR_BOOL,
  TENSOR_INT,
  TENSOR_FLOAT,
  GRAPH,

};

static const char *dtype_str[] = {
    "int8_t",      /* a */
    "bool",        /* b */
    "char",        /* c */
    "double",      /* d */
    "long double", /* e */
    "float",       /* f */
    "",            /* g */
    "uint8_t",     /* h */
    "int32_t",     /* i */
    "uint32_t",    /* j */
    "",            /* k */
    "int64_t",     /* l */
    "uint64_t",    /* m */
    "",            /* n */
    "",            /* o */
    "",            /* p */
    "",            /* q */
    "",            /* r */
    "int16_t",     /* s */
    "uint16_t",    /* t */
    "",            /* u */
    "",            /* v */
    "",            /* w */
    "",            /* x */
    "",            /* y */
    ""             /* z */
};

inline const char *dTypeName(int8_t v) { return dtype_str[0]; }
inline const char *dTypeName(bool v) { return dtype_str[1]; }
inline const char *dTypeName(char v) { return dtype_str[2]; }
inline const char *dTypeName(double v) { return dtype_str[3]; }
inline const char *dTypeName(long double v) { return dtype_str[4]; }
inline const char *dTypeName(float v) { return dtype_str[5]; }
inline const char *dTypeName(uint8_t v) { return dtype_str[7]; }
inline const char *dTypeName(int32_t v) { return dtype_str[8]; }
inline const char *dTypeName(uint32_t v) { return dtype_str[9]; }
inline const char *dTypeName(int64_t v) { return dtype_str[11]; }
inline const char *dTypeName(uint64_t v) { return dtype_str[12]; }
inline const char *dTypeName(int16_t v) { return dtype_str[18]; }
inline const char *dTypeName(uint16_t v) { return dtype_str[19]; }

template <typename T> DNNC_DataType getDNNC_DataType(T var) {
  std::string type_str = dTypeName(var);
  return getDNNC_DataType(type_str);
}

DNNC_DataType getDNNC_DataType(std::string stype);
std::string getDNNC_DataTypeStr(DNNC_DataType dtype);

IR_DataType getDNNC_IRType(std::string stype);
std::string getDNNC_IRTypeStr(IR_DataType dtype);

bool typePrecedence(DNNC_DataType ty1, DNNC_DataType ty2);

} // namespace dnnc
