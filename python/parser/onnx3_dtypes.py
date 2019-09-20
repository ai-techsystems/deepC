# Copyright 2018 The DNNC Authors. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
#
# This file is part of DNN compiler maintained at
# https://github.com/ai-techsystems/dnnCompiler
#

# TensorProto.DataType
# reference: onnx/onnx.proto3, line 319
UNDEFINED = 0;
FLOAT = 1;   # float
UINT8 = 2;   # uint8_t
INT8 = 3;    # int8_t
UINT16 = 4;  # uint16_t
INT16 = 5;   # int16_t
INT32 = 6;   # int32_t
INT64 = 7;   # int64_t
STRING = 8;  # string
BOOL = 9;    # bool

# IEEE754 half-precision floating-point format (16 bits wide).
# This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
FLOAT16 = 10;
DOUBLE = 11;
UINT32 = 12;
UINT64 = 13;
COMPLEX64 = 14;     # complex with float32 real and imaginary components
COMPLEX128 = 15;    # complex with float64 real and imaginary components

# Non-IEEE floating-point format based on IEEE754 single-precision
# floating-point number truncated to 16 bits.
# This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
BFLOAT16 = 16;

def type_INT():
    return [UINT8, INT8, UINT16, INT16, INT32]

def type_INT64():
    return INT64;

def type_FLOAT():
    return [FLOAT, FLOAT16, BFLOAT16]

def type_BOOL():
    return BOOL;

def type_COMPLEX():
    return [COMPLEX64, COMPLEX128]
