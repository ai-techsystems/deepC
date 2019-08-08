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


#include "core/datatypes.h"

namespace dnnc{

	DNNC_DataType getDNNC_DataType(std::string& stype)
	{
		DNNC_DataType type = DNNC_NOTYPE ;
		if ( stype == "float" )
			type = DNNC_FLOAT;
		if ( stype == "double" )
			type = DNNC_DOUBLE;
		if ( stype == "int32_t" )
			type = DNNC_INT32;
		if ( stype == "uint8_t" )
			type = DNNC_UINT8;
		if ( stype == "int16_t" )
			type = DNNC_INT16;
		if ( stype == "int8_t" )
			type = DNNC_INT8;
		if ( stype == "string" )
			type = DNNC_STRING;
		if ( stype == "int64_t" )
			type = DNNC_INT64;
		if ( stype == "bool" )
			type = DNNC_BOOL;
		if ( stype == "int8_t" )
			type = DNNC_INT8;
		if ( stype == "uint16_t" )
			type = DNNC_UINT16;
		if ( stype == "half" )
			type = DNNC_HALF;
		if ( stype == "uint32_t" )
			type = DNNC_UINT32;
		if ( stype == "int64_t" )
			type = DNNC_UINT64;
		return type;
	}
	template <typename T>
	DNNC_DataType getDNNC_DataType(T var)
	{
		DNNC_DataType type = DNNC_NOTYPE ;
		std::string  type_str = dtype_str[typeid(T).name()[0]-'a'];
		return getDNNC_DataType(type_str);

		return type;
	}
	std::string getDNNC_DataTypeStr(DNNC_DataType var)
	{
		std::string type = "";
		switch (var) 
		{
			case DNNC_FLOAT :
				type = "float" ;
				break;
			case DNNC_DOUBLE :
				type = "double" ;
				break;
			case DNNC_INT32 :
				type = "int32_t" ;
				break;
			case DNNC_UINT8 :
				type = "uint8_t" ;
				break;
			case DNNC_INT16 :
				type = "int16_t" ;
				break;
			case DNNC_INT8 :
				type = "int8_t" ;
				break;
			case DNNC_STRING :
				type = "string" ;
				break;
			case DNNC_INT64 :
				type = "int64_t" ;
				break;
			case DNNC_BOOL :
				type = "bool" ;
				break;
			case DNNC_QINT8 :// Quantized int8
				type = "int8_t" ;
				break;
			case DNNC_QUINT8 :// Quantized uint8
				type = "uint8_t" ;
				break;
			case DNNC_QINT32 :// Quantized int32
				type = "int32_t" ;
				break;
			case DNNC_BFLOAT16 :// Float32 truncated to 16 bits. 
				type = "half" ;
				break;
			case DNNC_QINT16 :// Quantized int16
				type = "int16_t" ;
				break;
			case DNNC_QUINT16 :// Quantized uint16
				type = "uint16_t" ;
				break;
			case DNNC_UINT16 :
				type = "uint16_t" ;
				break;
			case DNNC_HALF :
				type = "half" ;
				break;
			case DNNC_UINT32 :
				type = "uint32_t" ;
				break;
			case DNNC_UINT64 :
				type = "int64_t" ;
				break;
			case DNNC_RESOURCE :
				type = "resource" ;
				break;
			case DNNC_VARIANT :
				type = "variant" ;
				break;
			case DNNC_NOTYPE:
			default:
				type = "notype" ;
				break;
		}
		return type;
	}
}

