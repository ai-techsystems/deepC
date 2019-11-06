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
    std::string direction = "RIGHT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }
  dnnc::tensor<int> __rshift__(int scalar) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt(); 
    dnnc::tensor<int> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    std::string direction = "RIGHT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }


    /*  Assignment Right Shift  */
  %pyinplaceoper(__irshift__, dnnc::tensor::__irshift__, binaryfunc, nb_inplace_rshift);
  dnnc::tensor<int> __irshift__(dnnc::tensor<int>& other) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt(); 
    std::string direction = "RIGHT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }
  dnnc::tensor<int> __irshift__(int scalar) {
    dnnc::tensor<int> other_int = (*$self).copy().asTypeInt();
    dnnc::tensor<int> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    std::string direction = "RIGHT";
    dnnc::BitShift<int> op("pythonOp", direction);
    return op.compute(other_int, other);
  }



// <\/>

%pythoncode %{

def __getitem__(self, index):

  def get_item_helper_int(item, axis):
    start = item
    stop = item+1
    step = 1
    return start, stop, step

  def get_item_helper_slice(index, axis):
    flag = 0   # to check if all index items are supported or not
    start = 0
    stop = self.shape()[axis]
    step = 1
    if type(item.start) == int:
      start = item.start
    elif str(type(item.start)).split("'")[1] == "NoneType":
      pass
    else:
      print("start of", type(item.start), "not supported!")
      flag = 1
    if type(item.stop) == int:
      stop = item.stop
    elif str(type(item.stop)).split("'")[1] == "NoneType":
      pass
    else:
      print("stop of", type(item.stop), "not supported!")
      flag = 1
    if type(item.step) == int:
      step = item.step
    elif str(type(item.step)).split("'")[1] == "NoneType":
      pass
    else:
      print("step of", type(item.step), "not supported!")
      flag = 1
    return start, stop, step, flag

  if str(type(index)).split("'")[1] == "int":
    axis = 0
    start, stop, step = get_item_helper_int(index, axis)
    start = array([start]).asTypeULong()
    stop = array([stop]).asTypeULong()
    axis = array([axis]).asTypeInt()
    step = array([step]).asTypeULong()
    if (self.rank() == 1):
      # return self.operator[](index)
      return slice(self, start, stop, axis, step).reshape(1)
    return slice(self, start, stop, axis, step).reshape(self.shape()[1:])

  elif str(type(index)).split("'")[1] == "slice":
    axis = 0
    start, stop, step, flag = get_item_helper_slice(index, axis)
    if flag:
      return dc.empty(0)
    start = array([start]).asTypeULong()
    stop = array([stop]).asTypeULong()
    axis = array([axis]).asTypeInt()
    step = array([step]).asTypeULong()
    return slice(self, start, stop, axis, step)

  elif str(type(index)).split("'")[1] == "ellipsis":
    return self.copy()
    
  elif isinstance(index, list) or isinstance(index, tuple):
    
    # checks if any float or bool or complex is not present 
    if any(isinstance(x,(bool,float,complex)) for x in index):
      print("Restrict to only integers as a slicing argument!")
      return dc.empty(0)
    
    start_list = []
    stop_list = []
    step_list = []
    axis_list = []
    axis = -1   # -1 for starting axis as 0 in the next loops
    reshape_counter = 0
    replace_start = replace_stop = 0   # replace ellipsis with slice methods by index

    if Ellipsis in index:
      if (index.count(Ellipsis) > 1):
        print(index.count(Ellipsis),"'Ellipsis' found, maximum 1 is supported!")
        return dc.empty(0)
      elif (index.count(Ellipsis) == 1):
        non_ellipsis_count = 0
        for item in index:
          if str(type(item)).split("'")[1] == "int" or str(type(item)).split("'")[1] == "slice":
            non_ellipsis_count += 1 
        # replace holds start and stop index which will be replaced by slice method in place of ellipsis
        replace_start = index.index(Ellipsis)
        replace_stop = replace_start + self.rank() - non_ellipsis_count
      else:
        print("Error occured while handling ellipsis!")
        return dc.empty(0)

    for item in index:
      axis += 1
      if str(type(item)).split("'")[1] == "ellipsis":
        while (axis >= replace_start and axis < replace_stop):
          start = 0
          stop = self.shape()[axis]
          step = 1
          start_list.append(start)
          stop_list.append(stop)
          step_list.append(step)
          axis_list.append(axis)
          axis += 1
        axis -= 1   # recovering from last axis increment
      elif str(type(item)).split("'")[1] == "int":
        start, stop, step = get_item_helper_int(item, axis)
        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)
        axis_list.append(axis)
        reshape_counter += 1
      elif str(type(item)).split("'")[1] == "slice":
        start, stop, step, flag = get_item_helper_slice(index, axis)
        if flag:
          return dc.empty(0)
        start_list.append(start)
        stop_list.append(stop)
        step_list.append(step)
        axis_list.append(axis)
      else:
        print("Doen't support", item , "of", type(item), "as a slicing argument!")
        return dc.empty(0)
    
    while (axis < self.rank()-1):
      axis += 1
      start = 0
      stop = self.shape()[axis]
      step = 1
      start_list.append(start)
      stop_list.append(stop)
      step_list.append(step)
      axis_list.append(axis)

    start_list = array(start_list).asTypeULong()
    stop_list = array(stop_list).asTypeULong()
    axis_list = array(axis_list).asTypeInt()
    step_list = array(step_list).asTypeULong()
    result = slice(self, start_list, stop_list, axis_list, step_list)

    # This is a bug, we are returning 1D tensor, even if we need to return Scalar
    if (reshape_counter > 0):
      if (reshape_counter == self.rank()):
        # return result.operator[](0)
        return result.reshape(1)
      return (result).reshape(result.shape()[reshape_counter:])
    
    return result

  else :
    print("Doen't support", type(index), "as a slicing argument!")

  return dc.empty(0)


%}// <\/>

%pythoncode %{
def numpy(self) :
  """
     convert tensor to numpy array.
  """
  import numpy as np
  return np.array(self.data()).reshape(self.shape())  

def len(self):
  return self.length()

def __int__(self):
  return self.asTypeInt()

def __float__(self):
  return self.asTypeFloat()

def __bool__(self):
  return self.asTypeBool()
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
