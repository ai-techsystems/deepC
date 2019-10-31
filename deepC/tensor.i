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
%ignore *::operator==;
%ignore *::operator[];
%include "python/pyopers.swg"
%include "core/placeHolder.h"
%include "core/tensor.h"
%include "dnnc_swig_externs.h"
%{
#include <string>
#include <Python.h>
#include <object.h> // where Py_LT, Pt_GT are defined.
#include "core/tensor.h"
#include "operators/Add.h"
#include "operators/Abs.h"
#include "operators/And.h"
#include "operators/BitShift.h"
#include "operators/Equal.h"
#include "operators/FloorDiv.h"
#include "operators/Greater.h"
#include "operators/GreaterEqual.h"
#include "operators/Identity.h"
#include "operators/Less.h"
#include "operators/LessEqual.h"
#include "operators/Mod.h"
#include "operators/Mul.h"
#include "operators/Neg.h"
#include "operators/Not.h"
#include "operators/NotEqual.h"
#include "operators/Or.h"
#include "operators/Pow.h"
#include "operators/Remainder.h"
#include "operators/Sub.h"
#include "operators/TrueDiv.h"
#include "operators/Xor.h"
#include "dnnc_swig_externs.h"
extern std::vector<size_t> listTupleToVector_SizeT(PyObject *);
%}

%feature("python:slot",    "mp_subscript", functype="binaryfunc")             dnnc::tensor::__getitem__;
%feature("python:slot",    "mp_ass_subscript", functype="objobjargproc")      dnnc::tensor::__setitem__;
//%feature("python:slot",  "sq_slice", functype="ssizessizeargfunc")          __getslice__;
//%feature("python:slot",  "sq_ass_slice", functype="ssizessizeobjargproc")   __setslice__;
%feature("python:slot",    "sq_length", functype="lenfunc")                   dnnc::tensor::length;
%feature("python:slot",    "nb_bool", functype="inquiry")                     dnnc::tensor::operator bool;
%feature("python:slot",    "tp_repr", functype="reprfunc")                    dnnc::tensor::__str__;
%feature("python:slot",    "tp_str", functype="reprfunc")                     dnnc::tensor::__repr__;

//%implicitconv dnnc::tensor;
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
  bool __bool__() const {
    return (self->length());
  }
  std::string __str__() {
    return $self->to_string();
  }
  std::string __repr__() {
    return $self->to_string();
  }
  /* unary operators */
  %pybinoperator(__neg__, dnnc::tensor::__neg__, unaryfunc, nb_negative);
  dnnc::tensor<T> __neg__() {
    dnnc::Neg<T> op("pythonOp");
    return op.compute(*$self);
  }
  %pybinoperator(__pos__, dnnc::tensor::__pos__, unaryfunc, nb_positive);
  dnnc::tensor<T> __pos__() {
    return *$self;
  }
  %pybinoperator(__abs__, dnnc::tensor::__abs__, unaryfunc, nb_absolute);
  dnnc::tensor<T> __abs__() {
    dnnc::Abs<T> op("pythonOp");
    return op.compute(*$self);
  }
  %pybinoperator(__invert__, dnnc::tensor::__invert__, unaryfunc, nb_invert);
  dnnc::tensor<bool> __invert__() {
    dnnc::Not<bool, T> op("pythonOp");
    return op.compute(*$self);
  }


  /*  Binary TrueDiv */
  %pybinoperator(__truediv__, dnnc::tensor::__truediv__, binaryfunc, nb_truediv);
  dnnc::tensor<float> __truediv__(dnnc::tensor<T>& other) {
    return dnnc::true_div(*$self, other).asType<float>();
  }
  dnnc::tensor<float> __truediv__(PyObject *scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<float>;
  }
  
  dnnc::tensor<T> other(std::vector<size_t>(1,1));
  other.load(&data);
  
  return dnnc::true_div(*$self, other).asType<float>();
  }
  // 'swig -builtin' option limits all reverse operator from being overloaded.
  //       y=1+x; #(whre x and y are tensors) will not work
  %pybinoperator(__rtruediv__, dnnc::tensor::__rtruediv__, binaryfunc, nb_rdiv);
  dnnc::tensor<float> __rtruediv__(PyObject* scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<float>;
  }
  
  dnnc::tensor<T> other(std::vector<size_t>(1,1));
  other.load(&data);
  
  return dnnc::true_div(other, *$self).asType<float>();
  }



    /*  Assignment TrueDiv  */
  %pyinplaceoper(__itruediv__, dnnc::tensor::__itruediv__, binaryfunc, nb_inplace_truediv);
  dnnc::tensor<float> __itruediv__(dnnc::tensor<T>& other) {
    return dnnc::true_div(*$self, other).asType<float>();
  }
  dnnc::tensor<float> __itruediv__(T scalar) {
    dnnc::tensor<T> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    return dnnc::true_div(*$self, other).asType<float>();
  }



  /*  Binary FloorDiv */
  %pybinoperator(__floordiv__, dnnc::tensor::__floordiv__, binaryfunc, nb_floordiv);
  dnnc::tensor<int> __floordiv__(dnnc::tensor<T>& other) {
    return dnnc::floor_div(*$self, other).asType<int>();
  }
  dnnc::tensor<int> __floordiv__(PyObject *scalar) {
    T data ;
    if (PyBool_Check(scalar)) {
      data = scalar == Py_True ? true : false ;
    } else if (PyLong_Check(scalar)) {
      data = PyLong_AsLong(scalar);
    } else if (PyFloat_Check(scalar)) {
      data = PyFloat_AsDouble(scalar);
    } else {
      throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
      return dnnc::NULL_TENSOR<int>;
    }
  
    dnnc::tensor<T> other(std::vector<size_t>(1,1));
    other.load(&data);
    
    return dnnc::floor_div(*$self, other).asType<int>();
  }
  // 'swig -builtin' option limits all reverse operator from being overloaded.
  //       y=1+x; #(whre x and y are tensors) will not work
  %pybinoperator(__rfloordiv__, dnnc::tensor::__rfloordiv__, binaryfunc, nb_rfloordiv);
  dnnc::tensor<int> __rfloordiv__(PyObject* scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<int>;
  }
  
  dnnc::tensor<T> other(std::vector<size_t>(1,1));
  other.load(&data);
  
  return dnnc::floor_div(other, *$self).asType<int>();
  }



    /*  Assignment FloorDiv  */
  %pyinplaceoper(__ifloordiv__, dnnc::tensor::__ifloordiv__, binaryfunc, nb_inplace_floordiv);
  dnnc::tensor<int> __ifloordiv__(dnnc::tensor<T>& other) {
    return dnnc::floor_div(*$self, other).asType<int>();
  }
  dnnc::tensor<int> __ifloordiv__(T scalar) {
    dnnc::tensor<T> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    return dnnc::floor_div(*$self, other).asType<int>();
  }


    /*  Binary Left Shift  */
  %pyinplaceoper(__lshift__, dnnc::tensor::__lshift__, binaryfunc, nb_lshift);
  dnnc::tensor<int> __lshift__(dnnc::tensor<int>& other) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt(); 
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }
  dnnc::tensor<int> __lshift__(int scalar) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    dnnc::tensor<int> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }


    /*  Assignment Left Shift  */
  %pyinplaceoper(__ilshift__, dnnc::tensor::__ilshift__, binaryfunc, nb_inplace_lshift);
  dnnc::tensor<int> __ilshift__(dnnc::tensor<int>& other) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt(); 
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }
  dnnc::tensor<int> __ilshift__(int scalar) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    dnnc::tensor<int> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }



    /*  Binary Right Shift  */
  %pyinplaceoper(__rshift__, dnnc::tensor::__rshift__, binaryfunc, nb_rshift);
  dnnc::tensor<int> __rshift__(dnnc::tensor<int>& other) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt(); 
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }
  dnnc::tensor<int> __rshift__(int scalar) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt(); 
    dnnc::tensor<int> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }


    /*  Assignment Right Shift  */
  %pyinplaceoper(__irshift__, dnnc::tensor::__irshift__, binaryfunc, nb_inplace_rshift);
  dnnc::tensor<int> __irshift__(dnnc::tensor<int>& other) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt(); 
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }
  dnnc::tensor<int> __irshift__(int scalar) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    dnnc::tensor<int> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }



// <\/>// <\/>

%pythoncode %{
def numpy(self) :
  """
     convert tensor to numpy array.
  """
  import numpy as np
  return np.array(self.data()).reshape(self.shape())  

def len(self):
  return self.length()
%}


%feature("shadow") reshape(std::vector<size_t> &shape) %{
def reshape(self, *args):
  """
    reshape tensor to a new shape permitting same length of old and new shape.
  """
  new_shape = []
  for arg in args:
    if isinstance(arg, int):
      new_shape.append(arg)
    else:
      return $action (self, vectorSizeT(arg))
 
  if len(new_shape):
    return $action (self, vectorSizeT(new_shape))

  return $action (args)
%}

}
%template(boolplaceHolder)   dnnc::placeHolder<bool>;
%template(intplaceHolder)    dnnc::placeHolder<int>;
%template(uLongplaceHolder)  dnnc::placeHolder<size_t>;
%template(floatplaceHolder)  dnnc::placeHolder<float>;
%template(doubleplaceHolder) dnnc::placeHolder<double>;
%template(boolTensor)   dnnc::tensor<bool>;
%template(intTensor)    dnnc::tensor<int>;
%template(uLongTensor)  dnnc::tensor<size_t>;
%template(floatTensor)  dnnc::tensor<float>;
%template(doubleTensor) dnnc::tensor<double>;
namespace std {
  %template(vectorTensorBool) vector<dnnc::tensor<bool> >;
  %template(vectorTensorInt) vector<dnnc::tensor<int> >;
  %template(vectorTensorSizeT) vector<dnnc::tensor<size_t> >;
  %template(vectorTensorFloat) vector<dnnc::tensor<float> >;
}
