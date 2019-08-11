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
template <typename T> class Hardmax : public baseOperator<T> {
public:
  Hardmax(std::string name = "opHardmax", opAttributes *attrs = 0x0)
      : baseOperator<T>(opHardmax, name, attrs) {}
      tensor<T>
      compute(tensor<T>& a,int axis=0)
      {
      tensor<T> result(a.shape()[0], a.shape()[1]);
      Eigen::MatrixXf::Index max_index;
      DNNC_EIGEN_MATRIX(eigenMatrix1, a) ;
      
      if(!axis)
      {
        for(int i=0;i<int(a.shape()[0]);i++)
        {
          eigenMatrix1.row(i).maxCoeff(&max_index);
          for(int j=0;j<int(a.shape()[1]);j++)
          {
            if(j==max_index)
              eigenMatrix1(i,j)=1;
            else
              eigenMatrix1(i,j)=0;
          }
        }
      }
      else
      {
        for(int j=0;j<int(a.shape()[1]);j++)
        {
          eigenMatrix1.col(j).maxCoeff(&max_index);
          for(int i=0;i<int(a.shape()[0]);i++)
          {
            if(i==max_index)
              eigenMatrix1(i,j)=1;
            else
              eigenMatrix1(i,j)=0;
          }
        }
      }
      Matrix<T, Dynamic, Dynamic> eResult = eigenMatrix1 ;

      result.load( eResult.data() );

      return result;
      }
    };
} // namespace dnnc
