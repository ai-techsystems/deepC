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
// tensor.i
//

%ignore *::operator=;
%ignore *::operator[];
%include "core/tensor.h"
%{
#include <core/tensor.h>
#include <operators/Add.h>
extern std::vector<size_t> listTupleToVector_SizeT(PyObject *);
%}

%feature("python:slot", "mp_subscript", functype="binaryfunc") dnnc::tensor::__getitem__;
%feature("python:slot", "mp_ass_subscript", functype="objobjargproc") dnnc::tensor::__setitem__;
%extend dnnc::tensor {
  const T& __getitem__(PyObject* indices) {
    std::vector<size_t> vIndices = listTupleToVector_SizeT(indices);
    const T& item = $self->operator()(vIndices);
    return item;
  }
  void __setitem__(PyObject* indices, const T& data) {
    std::vector<size_t> vIndices = listTupleToVector_SizeT(indices);
    $self->load(data, vIndices);
    return ;
  }
  /* binary operators */
  dnnc::tensor<T> __add__(dnnc::tensor<T>& other) {
    dnnc::Add<T> op("pythonOp");
    return op.compute(*$self, other);
  }
  /* assignment operators */
  dnnc::tensor<T> __iadd__(dnnc::tensor<T>& other) {
    dnnc::Add<T> op("pythonOp");
    return op.compute(*$self, other);
  }
}
%template(bTensor) dnnc::tensor<bool>;
%template(iTensor) dnnc::tensor<int>;
%template(fTensor) dnnc::tensor<float>;
%template(dTensor) dnnc::tensor<double>;
namespace std {
  %template(itvec) vector<dnnc::tensor<int> >;
  %template(ftvec) vector<dnnc::tensor<float> >;
}

%pythoncode %{
    def astype(self, newType):
      if ( newType == "double" ) :
        return self.asTypeDouble();
      elif ( newType == "float" ) :
        return self.asTypeFloat();
      elif ( newType == "int" ) :
        return self.asTypeInt();
      elif ( newType == "bool" ) :
        return self.asTypeBool();
      else:
        raise ValueError("unsupported data type {} \n".format(newType))
      
      return self

    bTensor.astype = astype;
    iTensor.astype = astype;
    fTensor.astype = astype;
    dTensor.astype = astype;
%}
