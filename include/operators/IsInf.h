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
template <typename T> class IsInf : public baseOperator<T> {
public:
  IsInf(std::string name = "opIsInf", opAttributes *attrs = 0x0)
      : baseOperator<T>(opIsInf, name, attrs) {}
      static bool Is_INF(T x,int detect_negative,int detect_positive) {
         if( std::isinf(x) )
         {
           if( (x<0)&&(detect_negative) )
            return true;
           else if( (x>0)&&(detect_positive) )
            return true;
           else
            return false;
         }
         else
           return false;
       }
      // NOT GOOD to return by value
      tensor<bool>
      compute(tensor<T>& a,int detect_negative=1,int detect_positive=1)
      {
      tensor<bool> result(a.shape()[0], a.shape()[1]);

      DNNC_EIGEN_MATRIX(eigenMatrix1, a) ;

      auto c0 = std::bind(Is_INF, std::placeholders::_1, detect_negative,detect_positive);

      Matrix<bool, Dynamic, Dynamic> eResult=eigenMatrix1.unaryExpr(c0);
      result.load( eResult.data() );

      return result;
      }
};
} // namespace dnnc
