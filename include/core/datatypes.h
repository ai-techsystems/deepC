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

namespace dnnc {
	typedef enum DNNC_DataType {
		DNNC_FLOAT = 1,
		DNNC_DOUBLE = 2,
		DNNC_INT32 = 3,
		DNNC_UINT8 = 4,
		DNNC_INT16 = 5,
		DNNC_INT8 = 6,
		DNNC_STRING = 7,
		DNNC_INT64 = 9,
		DNNC_BOOL = 10,
		DNNC_QINT8 = 11,     // Quantized int8
		DNNC_QUINT8 = 12,    // Quantized uint8
		DNNC_QINT32 = 13,    // Quantized int32
		DNNC_BFLOAT16 = 14,  // Float32 truncated to 16 bits. 
		DNNC_QINT16 = 15,    // Quantized int16
		DNNC_QUINT16 = 16,   // Quantized uint16
		DNNC_UINT16 = 17,
		DNNC_HALF = 19,
		DNNC_UINT32 = 20,
		DNNC_UINT64 = 21,
		DNNC_RESOURCE = 22,
		DNNC_VARIANT = 23,
	} DNNC_DataType;

	static std::string dtype_str[] = { \
		"int8_t", /* a */
		"", /* b */
		"char", /* c */
		"double", /* d */
		"long double", /* e */
		"float", /* f */
		"", /* g */
		"uint8_t", /* h */
		"int32_t", /* i */
		"uint32_t", /* j */
		"", /* k */
		"int64_t", /* l */
		"uint64_t", /* m */
		"", /* n */
		"", /* o */
		"", /* p */
		"", /* q */
		"", /* r */
		"int16_t", /* s */
		"uint16_t", /* t */
		"", /* u */
		"", /* v */
		"", /* w */
		"", /* x */
		"", /* y */
		"" /* z */
};
}
