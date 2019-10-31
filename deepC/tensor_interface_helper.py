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

# This file is required by ../dnnCompiler/swig/op_gen.py


def tensor_swig_helper_div():
  
  # true-div and floor-div are special cases that we are just copying over
	
  s = '''
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
'''
  return s


def tensor_swig_helper_comparison(dc_operator, operator_header, operator_python):

  s = '''
  /*  Comparison <operator>  */
    %pycompare(__<operand>__, dnnc::tensor::__<operand>__, Py_<operand_upper>);
  dnnc::tensor<bool> __<operand>__(dnnc::tensor<T>& other) {
    return dnnc::<dc_operator>(*$self, other).asType<bool>();
  }
'''
  s = s.replace("<operator>",operator_header).replace("<dc_operator>",dc_operator).replace("<operand>",operator_python).replace("<operand_upper>",operator_python.upper()) + "\n\n"
  return s


def tensor_swig_helper_logical(dc_operator, operator_header, operator_python):

	s = '''
  /*  Logical <operator>  */
  %pybinoperator(__<operand>__, dnnc::tensor::__<operand>__, binaryfunc, nb_<operand>);
  dnnc::tensor<bool> __<operand>__(dnnc::tensor<T>& other) {
    return dnnc::<dc_operator>(*$self, other).asType<bool>();
  }
  dnnc::tensor<bool> __<operand>__(T scalar) {
    dnnc::tensor<T> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    return dnnc::<dc_operator>(*$self, other).asType<bool>();
  }
  %pybinoperator(__r<operand>__, dnnc::tensor::__r<operand>__, binaryfunc, nb_r<operand>);
  dnnc::tensor<bool> __r<operand>__(T scalar) {
    dnnc::tensor<T> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    return dnnc::<dc_operator>(other, *$self).asType<bool>();
  }



  /*  Assignment <operator>  */
  %pyinplaceoper(__i<operand>__, dnnc::tensor::__i<operand>__, binaryfunc, nb_inplace_<operand>);
  dnnc::tensor<bool> __i<operand>__(dnnc::tensor<T>& other) {
    return dnnc::<dc_operator>(*$self, other).asType<bool>();
  }
  dnnc::tensor<bool> __i<operand>__(T scalar) {
    dnnc::tensor<T> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    return dnnc::<dc_operator>(*$self, other).asType<bool>();
  }
'''
	s = s.replace("<operator>",operator_header).replace("<operand>",operator_python).replace("<dc_operator>",dc_operator) + "\n\n"
	return s


def tensor_swig_helper_binary(dc_operator, operator_header, operator_python):

	s = '''
  /*  Binary <operator>  */
  %pybinoperator(__<operand>__, dnnc::tensor::__<operand>__, binaryfunc, nb_<operand>);
  dnnc::tensor<T> __<operand>__(dnnc::tensor<bool>& other) {
  return dnnc::<dc_operator>(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __<operand>__(dnnc::tensor<int>& other) {
  return dnnc::<dc_operator>(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __<operand>__(dnnc::tensor<size_t>& other) {
  return dnnc::<dc_operator>(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __<operand>__(dnnc::tensor<float>& other) {
  return dnnc::<dc_operator>(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __<operand>__(PyObject *scalar) {
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

  dnnc::tensor<T> other(std::vector<size_t>(1,1));
  other.load(&data);

  return dnnc::<dc_operator>(*$self, other).asType<T>();
  }
  // 'swig -builtin' option limits all reverse operator from being overloaded.
  //       y=1+x; #(whre x and y are tensors) will not work
  %pybinoperator(__r<operand>__, dnnc::tensor::__r<operand>__, binaryfunc, nb_r<operand>);
  dnnc::tensor<T> __r<operand>__(PyObject* scalar) {
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

  dnnc::tensor<T> other(std::vector<size_t>(1,1));
  other.load(&data);

  return dnnc::<dc_operator>(other, *$self).asType<T>();
  }



    /*  Assignment <operator>  */
  %pyinplaceoper(__i<operand>__, dnnc::tensor::__i<operand>__, binaryfunc, nb_inplace_<operand>);
  dnnc::tensor<T> __i<operand>__(dnnc::tensor<T>& other) {
    return dnnc::<dc_operator>(*$self, other).asType<T>();
  }
  dnnc::tensor<T> __i<operand>__(T scalar) {
    dnnc::tensor<T> other(std::vector<size_t>(1,1));
    other.load(&scalar);
    return dnnc::<dc_operator>(*$self, other).asType<T>();
  }
'''
	s = s.replace("<operator>",operator_header).replace("<operand>",operator_python).replace("<dc_operator>",dc_operator) + "\n\n"
	return s

