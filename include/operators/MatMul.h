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
#include "operators/baseOperator.h"
#include <string>

using namespace Eigen;

namespace dnnc {
  template <typename T>
  class MatMul : public baseOperator<T> {
    protected:
    public:
      MatMul(std::string name="opMatMul", opAttributes* attrs=0x0) : 
	     baseOperator<T>(opMatMul, name, attrs)
      {}
	  // NOT GOOD to return by value
      tensor<T> 
      compute(tensor<T>& a, tensor<T>& b)
	  {
		  if ( a.rank() != 2 || b.rank() != 2 ) 
			  throw std::invalid_argument("invalid tensor rank."); 

		  if ( a.shape()[1] != b.shape()[0] )
			  throw std::invalid_argument("tensor dimenions not appropriate for multiplication operator."); 
		  
		  tensor<T> result(a.shape()[0], b.shape()[1]); 
		  
		  DNNC_EIGEN_MATRIX(eigenMatrixA, a) ; 
		  DNNC_EIGEN_MATRIX(eigenMatrixB, b) ; 

		  Matrix<T, Dynamic, Dynamic> eResult = eigenMatrixA * eigenMatrixB ; 
		  
		  result.load( eResult.data() ); 

		  return result;
	  }
  };
}
