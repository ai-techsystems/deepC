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
  
  dnnc::tensor<T> other(1);
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
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::true_div(other, *$self).asType<float>();
  }



    /*  Assignment TrueDiv  */
  %pyinplaceoper(__itruediv__, dnnc::tensor::__itruediv__, binaryfunc, nb_inplace_truediv);
  dnnc::tensor<float> __itruediv__(dnnc::tensor<T>& other) {
    return dnnc::true_div(*$self, other).asType<float>();
  }
  dnnc::tensor<float> __itruediv__(T scalar) {
    dnnc::tensor<T> other(1);
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
  
    dnnc::tensor<T> other(1);
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
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::floor_div(other, *$self).asType<int>();
  }



    /*  Assignment FloorDiv  */
  %pyinplaceoper(__ifloordiv__, dnnc::tensor::__ifloordiv__, binaryfunc, nb_inplace_floordiv);
  dnnc::tensor<int> __ifloordiv__(dnnc::tensor<T>& other) {
    return dnnc::floor_div(*$self, other).asType<int>();
  }
  dnnc::tensor<int> __ifloordiv__(T scalar) {
    dnnc::tensor<T> other(1);
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
    dnnc::tensor<int> other(1);
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
    dnnc::tensor<int> other(1);
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
    dnnc::tensor<int> other(1);
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
    dnnc::tensor<int> other(1);
    other.load(&scalar);
    std::string direction = "LEFT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }



// <\/>
  /*  Binary Add  */
  %pybinoperator(__add__, dnnc::tensor::__add__, binaryfunc, nb_add);
  dnnc::tensor<T> __add__(dnnc::tensor<bool>& other) {
  return dnnc::add(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __add__(dnnc::tensor<int>& other) {
  return dnnc::add(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __add__(dnnc::tensor<size_t>& other) {
  return dnnc::add(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __add__(dnnc::tensor<float>& other) {
  return dnnc::add(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __add__(PyObject *scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<T>;
  }
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::add(*$self, other).asType<T>();
  }
  // 'swig -builtin' option limits all reverse operator from being overloaded.
  //       y=1+x; #(whre x and y are tensors) will not work
  %pybinoperator(__radd__, dnnc::tensor::__radd__, binaryfunc, nb_radd);
  dnnc::tensor<T> __radd__(PyObject* scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<T>;
  }
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::add(other, *$self).asType<T>();
  }



    /*  Assignment Add  */
  %pyinplaceoper(__iadd__, dnnc::tensor::__iadd__, binaryfunc, nb_inplace_add);
  dnnc::tensor<T> __iadd__(dnnc::tensor<T>& other) {
    return dnnc::add(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __iadd__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::add(*$self, other).asType<T>();
  }



  /*  Binary Sub  */
  %pybinoperator(__sub__, dnnc::tensor::__sub__, binaryfunc, nb_sub);
  dnnc::tensor<T> __sub__(dnnc::tensor<bool>& other) {
  return dnnc::sub(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __sub__(dnnc::tensor<int>& other) {
  return dnnc::sub(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __sub__(dnnc::tensor<size_t>& other) {
  return dnnc::sub(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __sub__(dnnc::tensor<float>& other) {
  return dnnc::sub(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __sub__(PyObject *scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<T>;
  }
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::sub(*$self, other).asType<T>();
  }
  // 'swig -builtin' option limits all reverse operator from being overloaded.
  //       y=1+x; #(whre x and y are tensors) will not work
  %pybinoperator(__rsub__, dnnc::tensor::__rsub__, binaryfunc, nb_rsub);
  dnnc::tensor<T> __rsub__(PyObject* scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<T>;
  }
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::sub(other, *$self).asType<T>();
  }



    /*  Assignment Sub  */
  %pyinplaceoper(__isub__, dnnc::tensor::__isub__, binaryfunc, nb_inplace_sub);
  dnnc::tensor<T> __isub__(dnnc::tensor<T>& other) {
    return dnnc::sub(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __isub__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::sub(*$self, other).asType<T>();
  }



  /*  Binary Mul  */
  %pybinoperator(__mul__, dnnc::tensor::__mul__, binaryfunc, nb_mul);
  dnnc::tensor<T> __mul__(dnnc::tensor<bool>& other) {
  return dnnc::mul(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __mul__(dnnc::tensor<int>& other) {
  return dnnc::mul(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __mul__(dnnc::tensor<size_t>& other) {
  return dnnc::mul(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __mul__(dnnc::tensor<float>& other) {
  return dnnc::mul(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __mul__(PyObject *scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<T>;
  }
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::mul(*$self, other).asType<T>();
  }
  // 'swig -builtin' option limits all reverse operator from being overloaded.
  //       y=1+x; #(whre x and y are tensors) will not work
  %pybinoperator(__rmul__, dnnc::tensor::__rmul__, binaryfunc, nb_rmul);
  dnnc::tensor<T> __rmul__(PyObject* scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<T>;
  }
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::mul(other, *$self).asType<T>();
  }



    /*  Assignment Mul  */
  %pyinplaceoper(__imul__, dnnc::tensor::__imul__, binaryfunc, nb_inplace_mul);
  dnnc::tensor<T> __imul__(dnnc::tensor<T>& other) {
    return dnnc::mul(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __imul__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::mul(*$self, other).asType<T>();
  }



  /*  Binary Remainder  */
  %pybinoperator(__mod__, dnnc::tensor::__mod__, binaryfunc, nb_mod);
  dnnc::tensor<T> __mod__(dnnc::tensor<bool>& other) {
  return dnnc::remainder(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __mod__(dnnc::tensor<int>& other) {
  return dnnc::remainder(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __mod__(dnnc::tensor<size_t>& other) {
  return dnnc::remainder(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __mod__(dnnc::tensor<float>& other) {
  return dnnc::remainder(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __mod__(PyObject *scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<T>;
  }
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::remainder(*$self, other).asType<T>();
  }
  // 'swig -builtin' option limits all reverse operator from being overloaded.
  //       y=1+x; #(whre x and y are tensors) will not work
  %pybinoperator(__rmod__, dnnc::tensor::__rmod__, binaryfunc, nb_rmod);
  dnnc::tensor<T> __rmod__(PyObject* scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<T>;
  }
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::remainder(other, *$self).asType<T>();
  }



    /*  Assignment Remainder  */
  %pyinplaceoper(__imod__, dnnc::tensor::__imod__, binaryfunc, nb_inplace_mod);
  dnnc::tensor<T> __imod__(dnnc::tensor<T>& other) {
    return dnnc::remainder(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __imod__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::remainder(*$self, other).asType<T>();
  }



  /*  Binary Pow  */
  %pybinoperator(__pow__, dnnc::tensor::__pow__, binaryfunc, nb_pow);
  dnnc::tensor<T> __pow__(dnnc::tensor<bool>& other) {
  return dnnc::power(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __pow__(dnnc::tensor<int>& other) {
  return dnnc::power(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __pow__(dnnc::tensor<size_t>& other) {
  return dnnc::power(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __pow__(dnnc::tensor<float>& other) {
  return dnnc::power(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __pow__(PyObject *scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<T>;
  }
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::power(*$self, other).asType<T>();
  }
  // 'swig -builtin' option limits all reverse operator from being overloaded.
  //       y=1+x; #(whre x and y are tensors) will not work
  %pybinoperator(__rpow__, dnnc::tensor::__rpow__, binaryfunc, nb_rpow);
  dnnc::tensor<T> __rpow__(PyObject* scalar) {
  T data ;
  if (PyBool_Check(scalar)) {
    data = scalar == Py_True ? true : false ;
  } else if (PyLong_Check(scalar)) {
    data = PyLong_AsLong(scalar);
  } else if (PyFloat_Check(scalar)) {
    data = PyFloat_AsDouble(scalar);
  } else {
    throw std::invalid_argument(std::string("scalar operation not supported with tensor type <") + dnnc::dtype_str[typeid(T).name()[0] - 'a'] + std::string(">") );
    return dnnc::NULL_TENSOR<T>;
  }
  
  dnnc::tensor<T> other(1);
  other.load(&data);
  
  return dnnc::power(other, *$self).asType<T>();
  }



    /*  Assignment Pow  */
  %pyinplaceoper(__ipow__, dnnc::tensor::__ipow__, binaryfunc, nb_inplace_pow);
  dnnc::tensor<T> __ipow__(dnnc::tensor<T>& other) {
    return dnnc::power(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __ipow__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::power(*$self, other).asType<T>();
  }



  /*  Logical And  */
  %pybinoperator(__and__, dnnc::tensor::__and__, binaryfunc, nb_and);
  dnnc::tensor<bool> __and__(dnnc::tensor<T>& other) {
    return dnnc::logical_and(*$self, other).asType<bool>();
  }
  dnnc::tensor<bool> __and__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::logical_and(*$self, other).asType<bool>();
  }
  %pybinoperator(__rand__, dnnc::tensor::__rand__, binaryfunc, nb_rand);
  dnnc::tensor<bool> __rand__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::logical_and(other, *$self).asType<bool>();
  }



  /*  Assignment And  */
  %pyinplaceoper(__iand__, dnnc::tensor::__iand__, binaryfunc, nb_inplace_and);
  dnnc::tensor<bool> __iand__(dnnc::tensor<T>& other) {
    return dnnc::logical_and(*$self, other).asType<bool>();
  }
  dnnc::tensor<bool> __iand__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::logical_and(*$self, other).asType<bool>();
  }



  /*  Logical Or  */
  %pybinoperator(__or__, dnnc::tensor::__or__, binaryfunc, nb_or);
  dnnc::tensor<bool> __or__(dnnc::tensor<T>& other) {
    return dnnc::logical_or(*$self, other).asType<bool>();
  }
  dnnc::tensor<bool> __or__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::logical_or(*$self, other).asType<bool>();
  }
  %pybinoperator(__ror__, dnnc::tensor::__ror__, binaryfunc, nb_ror);
  dnnc::tensor<bool> __ror__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::logical_or(other, *$self).asType<bool>();
  }



  /*  Assignment Or  */
  %pyinplaceoper(__ior__, dnnc::tensor::__ior__, binaryfunc, nb_inplace_or);
  dnnc::tensor<bool> __ior__(dnnc::tensor<T>& other) {
    return dnnc::logical_or(*$self, other).asType<bool>();
  }
  dnnc::tensor<bool> __ior__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::logical_or(*$self, other).asType<bool>();
  }



  /*  Logical Xor  */
  %pybinoperator(__xor__, dnnc::tensor::__xor__, binaryfunc, nb_xor);
  dnnc::tensor<bool> __xor__(dnnc::tensor<T>& other) {
    return dnnc::logical_xor(*$self, other).asType<bool>();
  }
  dnnc::tensor<bool> __xor__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::logical_xor(*$self, other).asType<bool>();
  }
  %pybinoperator(__rxor__, dnnc::tensor::__rxor__, binaryfunc, nb_rxor);
  dnnc::tensor<bool> __rxor__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::logical_xor(other, *$self).asType<bool>();
  }



  /*  Assignment Xor  */
  %pyinplaceoper(__ixor__, dnnc::tensor::__ixor__, binaryfunc, nb_inplace_xor);
  dnnc::tensor<bool> __ixor__(dnnc::tensor<T>& other) {
    return dnnc::logical_xor(*$self, other).asType<bool>();
  }
  dnnc::tensor<bool> __ixor__(T scalar) {
    dnnc::tensor<T> other(1);
    other.load(&scalar);
    return dnnc::logical_xor(*$self, other).asType<bool>();
  }



  /*  Comparison Equal  */
    %pycompare(__eq__, dnnc::tensor::__eq__, Py_EQ);
  dnnc::tensor<bool> __eq__(dnnc::tensor<T>& other) {
    return dnnc::equal(*$self, other).asType<bool>();
  }



  /*  Comparison Less  */
    %pycompare(__lt__, dnnc::tensor::__lt__, Py_LT);
  dnnc::tensor<bool> __lt__(dnnc::tensor<T>& other) {
    return dnnc::less(*$self, other).asType<bool>();
  }



  /*  Comparison Greater  */
    %pycompare(__gt__, dnnc::tensor::__gt__, Py_GT);
  dnnc::tensor<bool> __gt__(dnnc::tensor<T>& other) {
    return dnnc::greater(*$self, other).asType<bool>();
  }



  /*  Comparison NotEqual  */
    %pycompare(__ne__, dnnc::tensor::__ne__, Py_NE);
  dnnc::tensor<bool> __ne__(dnnc::tensor<T>& other) {
    return dnnc::not_equal(*$self, other).asType<bool>();
  }



  /*  Comparison LessEqual  */
    %pycompare(__le__, dnnc::tensor::__le__, Py_LE);
  dnnc::tensor<bool> __le__(dnnc::tensor<T>& other) {
    return dnnc::less_equal(*$self, other).asType<bool>();
  }



  /*  Comparison GreaterEqual  */
    %pycompare(__ge__, dnnc::tensor::__ge__, Py_GE);
  dnnc::tensor<bool> __ge__(dnnc::tensor<T>& other) {
    return dnnc::greater_equal(*$self, other).asType<bool>();
  }


// <\/>

}
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
