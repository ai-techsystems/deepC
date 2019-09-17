# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for divitional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License") you may not use this file except in compliance
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

import common

import dnnc as dc
import numpy as np
import unittest

class tensorAssignmentOperatorsTest(unittest.TestCase):

	def setUp(self):

		self.np_bool_0_4 = np.arange(5).astype(np.bool)
		self.np_bool_5_9 = np.arange(5,10).astype(np.bool)

		self.np_int_0_4 = np.arange(5).astype(np.int)
		self.np_int_5_9 = np.arange(5,10).astype(np.int)

		self.np_float_0_4 = np.arange(5).astype(np.float)
		self.np_float_5_9 = np.arange(5,10).astype(np.float)

		self.np_double_0_4 = np.arange(5).astype(np.double)
		self.np_double_5_9 = np.arange(5,10).astype(np.double)

		self.dc_bool_0_4 = dc.arange(5).asTypeBool()
		self.dc_bool_5_9 = dc.arange(5,10).asTypeBool()

		self.dc_int_0_4 = dc.arange(5).asTypeInt()
		self.dc_int_5_9 = dc.arange(5,10).asTypeInt()

		self.dc_float_0_4 = dc.arange(5).asTypeFloat()
		self.dc_float_5_9 = dc.arange(5,10).asTypeFloat()

		self.dc_double_0_4 = dc.arange(5).asTypeDouble()
		self.dc_double_5_9 = dc.arange(5,10).asTypeDouble()
		
	
	# Assignment Add
	
	# bool_tensor *= bool_scalar  
	def test_Assignment_Add_bool_tensor_bool_scalar (self):
		temp_np = self.np_bool_0_4.copy()
		temp_np *= True
		temp_dc = self.dc_bool_0_4.copy()
		temp_dc *= True
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# bool_tensor *= bool_tensor
	def test_Assignment_Add_bool_tensor_bool_tensor (self):
		temp_np = self.np_bool_0_4.copy()
		temp_np *= self.np_bool_5_9
		temp_dc = self.dc_bool_0_4.copy()
		temp_dc *= self.dc_bool_5_9
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# int_tensor *= bool_scalar
	def test_Assignment_Add_int_tensor_int_scalar (self):
		temp_np = self.np_int_0_4.copy()
		temp_np *= True
		temp_dc = self.dc_int_0_4.copy()
		temp_dc *= True
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor *= int_scalar
	def test_Assignment_Add_int_tensor_int_scalar (self):
		temp_np = self.np_int_0_4.copy()
		temp_np *= 5
		temp_dc = self.dc_int_0_4.copy()
		temp_dc *= 5
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor *= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Add_int_tensor_bool_tensor (self):
	# 	temp_np = self.np_int_0_4.copy()
	# 	temp_np *= self.np_bool_5_9
	# 	temp_dc = self.dc_int_0_4.copy()
	# 	temp_dc *= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor *= int_tensor
	def test_Assignment_Add_int_tensor_int_tensor (self):
		temp_np = self.np_int_0_4.copy()
		temp_np *= self.np_int_5_9
		temp_dc = self.dc_int_0_4.copy()
		temp_dc *= self.dc_int_5_9
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# float_tensor *= bool_scalar
	def test_Assignment_Add_float_tensor_bool_scalar (self):
		temp_np = self.np_float_0_4.copy()
		temp_np *= True
		temp_dc = self.dc_float_0_4.copy()
		temp_dc *= True
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor *= int_scalar
	def test_Assignment_Add_float_tensor_int_scalar (self):
		temp_np = self.np_float_0_4.copy()
		temp_np *= 5
		temp_dc = self.dc_float_0_4.copy()
		temp_dc *= 5
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor *= float_scalar
	def test_Assignment_Add_float_tensor_float_scalar (self):
		temp_np = self.np_float_0_4.copy()
		temp_np *= 5.0
		temp_dc = self.dc_float_0_4.copy()
		temp_dc *= 5.0
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor *= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Add_float_tensor_bool_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np *= self.np_bool_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc *= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor *= int_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Add_float_tensor_int_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np *= self.np_int_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc *= self.dc_int_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor *= float_tensor
	def test_Assignment_Add_float_tensor_float_tensor (self):
		temp_np = self.np_float_0_4.copy()
		temp_np *= self.np_float_5_9
		temp_dc = self.dc_float_0_4.copy()
		temp_dc *= self.dc_float_5_9
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# float_tensor *= double_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Add_float_tensor_double_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np *= self.np_double_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc *= self.dc_double_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	'''
	Double Add tensor operator fails on Dnnc
		x: array([ 5.,  7.,  9., 11., 13.])
		y: array(<Swig Object of type 'std::vector< double,std::allocator< double > > *' at 0x7f1e95c75f80>,dtype=object)
		
		test_Assignment_Add_double_tensor_double_tensor (__main__.tensorDetailedOperatorsTest) ... FAIL
		swig/python detected a memory leak of type 'std::vector< double,std::allocator< double > > *', no destructor found.

	'''
	# double_tensor *= bool_scalar
	# def test_Assignment_Add_double_tensor_bool_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= True
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= True
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor *= int_scalar
	# def test_Assignment_Add_double_tensor_int_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= 5
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= 5
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor *= float_scalar
	# def test_Assignment_Add_double_tensor_float_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= 5.0
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= 5.0
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor *= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Add_double_tensor_bool_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= self.np_bool_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor *= int_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Add_double_tensor_int_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= self.np_int_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= self.dc_int_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor *= float_tensor
	# def test_Assignment_Add_double_tensor_float_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= self.np_float_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= self.dc_float_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# double_tensor *= double_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Add_double_tensor_double_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= self.np_double_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= self.dc_double_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	

	# Assignment Sub
	'''
	Bool tensor with Assignment Sub operator works like Assignment Xor operator
	Numpy shows this: "numpy boolean subtract, the `-` operator, is deprecated, use 
				   the bitwise_xor, the `^` operator, or the logical_xor function instead."
	So skiping Bool tensor in Assignment Sub operator
	'''

	# int_tensor -= bool_scalar
	def test_Assignment_Sub_int_tensor_int_scalar (self):
		temp_np = self.np_int_0_4.copy()
		temp_np -= True
		temp_dc = self.dc_int_0_4.copy()
		temp_dc -= True
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor -= int_scalar
	def test_Assignment_Sub_int_tensor_int_scalar (self):
		temp_np = self.np_int_0_4.copy()
		temp_np -= 5
		temp_dc = self.dc_int_0_4.copy()
		temp_dc -= 5
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor -= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Sub_int_tensor_bool_tensor (self):
	# 	temp_np = self.np_int_0_4.copy()
	# 	temp_np -= self.np_bool_5_9
	# 	temp_dc = self.dc_int_0_4.copy()
	# 	temp_dc -= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor -= int_tensor
	def test_Assignment_Sub_int_tensor_int_tensor (self):
		temp_np = self.np_int_0_4.copy()
		temp_np -= self.np_int_5_9
		temp_dc = self.dc_int_0_4.copy()
		temp_dc -= self.dc_int_5_9
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# float_tensor -= bool_scalar
	def test_Assignment_Sub_float_tensor_bool_scalar (self):
		temp_np = self.np_float_0_4.copy()
		temp_np -= True
		temp_dc = self.dc_float_0_4.copy()
		temp_dc -= True
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor -= int_scalar
	def test_Assignment_Sub_float_tensor_int_scalar (self):
		temp_np = self.np_float_0_4.copy()
		temp_np -= 5
		temp_dc = self.dc_float_0_4.copy()
		temp_dc -= 5
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor -= float_scalar
	def test_Assignment_Sub_float_tensor_float_scalar (self):
		temp_np = self.np_float_0_4.copy()
		temp_np -= 5.0
		temp_dc = self.dc_float_0_4.copy()
		temp_dc -= 5.0
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor -= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Sub_float_tensor_bool_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np -= self.np_bool_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc -= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor -= int_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Sub_float_tensor_int_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np -= self.np_int_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc -= self.dc_int_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor -= float_tensor
	def test_Assignment_Sub_float_tensor_float_tensor (self):
		temp_np = self.np_float_0_4.copy()
		temp_np -= self.np_float_5_9
		temp_dc = self.dc_float_0_4.copy()
		temp_dc -= self.dc_float_5_9
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# float_tensor -= double_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Sub_float_tensor_double_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np -= self.np_double_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc -= self.dc_double_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	'''
	Double Sub tensor operator fails on Dnnc
		x: array([ 5.,  7.,  9., 11., 13.])
		y: array(<Swig Object of type 'std::vector< double,std::allocator< double > > *' at 0x7f1e95c75f80>,dtype=object)
		
		test_Assignment_Sub_double_tensor_double_tensor (__main__.tensorDetailedOperatorsTest) ... FAIL
		swig/python detected a memory leak of type 'std::vector< double,std::allocator< double > > *', no destructor found.

	'''
	# double_tensor -= bool_scalar
	# def test_Assignment_Sub_double_tensor_bool_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np -= True
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc -= True
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor -= int_scalar
	# def test_Assignment_Sub_double_tensor_int_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np -= 5
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc -= 5
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor -= float_scalar
	# def test_Assignment_Sub_double_tensor_float_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np -= 5.0
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc -= 5.0
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor -= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Sub_double_tensor_bool_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np -= self.np_bool_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc -= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor -= int_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Sub_double_tensor_int_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np -= self.np_int_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc -= self.dc_int_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor -= float_tensor
	# def test_Assignment_Sub_double_tensor_float_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np -= self.np_float_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc -= self.dc_float_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# double_tensor -= double_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Sub_double_tensor_double_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np -= self.np_double_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc -= self.dc_double_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	

	# Assignment Mul
	
	# bool_tensor *= bool_scalar  
	def test_Assignment_Mul_bool_tensor_bool_scalar (self):
		temp_np = self.np_bool_0_4.copy()
		temp_np *= True
		temp_dc = self.dc_bool_0_4.copy()
		temp_dc *= True
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# bool_tensor *= bool_tensor
	def test_Assignment_Mul_bool_tensor_bool_tensor (self):
		temp_np = self.np_bool_0_4.copy()
		temp_np *= self.np_bool_5_9
		temp_dc = self.dc_bool_0_4.copy()
		temp_dc *= self.dc_bool_5_9
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# int_tensor *= bool_scalar
	def test_Assignment_Mul_int_tensor_int_scalar (self):
		temp_np = self.np_int_0_4.copy()
		temp_np *= True
		temp_dc = self.dc_int_0_4.copy()
		temp_dc *= True
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor *= int_scalar
	def test_Assignment_Mul_int_tensor_int_scalar (self):
		temp_np = self.np_int_0_4.copy()
		temp_np *= 5
		temp_dc = self.dc_int_0_4.copy()
		temp_dc *= 5
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor *= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Mul_int_tensor_bool_tensor (self):
	# 	temp_np = self.np_int_0_4.copy()
	# 	temp_np *= self.np_bool_5_9
	# 	temp_dc = self.dc_int_0_4.copy()
	# 	temp_dc *= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor *= int_tensor
	def test_Assignment_Mul_int_tensor_int_tensor (self):
		temp_np = self.np_int_0_4.copy()
		temp_np *= self.np_int_5_9
		temp_dc = self.dc_int_0_4.copy()
		temp_dc *= self.dc_int_5_9
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# float_tensor *= bool_scalar
	def test_Assignment_Mul_float_tensor_bool_scalar (self):
		temp_np = self.np_float_0_4.copy()
		temp_np *= True
		temp_dc = self.dc_float_0_4.copy()
		temp_dc *= True
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor *= int_scalar
	def test_Assignment_Mul_float_tensor_int_scalar (self):
		temp_np = self.np_float_0_4.copy()
		temp_np *= 5
		temp_dc = self.dc_float_0_4.copy()
		temp_dc *= 5
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor *= float_scalar
	def test_Assignment_Mul_float_tensor_float_scalar (self):
		temp_np = self.np_float_0_4.copy()
		temp_np *= 5.0
		temp_dc = self.dc_float_0_4.copy()
		temp_dc *= 5.0
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor *= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Mul_float_tensor_bool_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np *= self.np_bool_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc *= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor *= int_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Mul_float_tensor_int_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np *= self.np_int_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc *= self.dc_int_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor *= float_tensor
	def test_Assignment_Mul_float_tensor_float_tensor (self):
		temp_np = self.np_float_0_4.copy()
		temp_np *= self.np_float_5_9
		temp_dc = self.dc_float_0_4.copy()
		temp_dc *= self.dc_float_5_9
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# float_tensor *= double_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Mul_float_tensor_double_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np *= self.np_double_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc *= self.dc_double_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	'''
	Double Mul tensor operator fails on Dnnc
		x: array([ 5.,  7.,  9., 11., 13.])
		y: array(<Swig Object of type 'std::vector< double,std::allocator< double > > *' at 0x7f1e95c75f80>,dtype=object)
		
		test_Assignment_Mul_double_tensor_double_tensor (__main__.tensorDetailedOperatorsTest) ... FAIL
		swig/python detected a memory leak of type 'std::vector< double,std::allocator< double > > *', no destructor found.

	'''
	# double_tensor *= bool_scalar
	# def test_Assignment_Mul_double_tensor_bool_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= True
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= True
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor *= int_scalar
	# def test_Assignment_Mul_double_tensor_int_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= 5
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= 5
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor *= float_scalar
	# def test_Assignment_Mul_double_tensor_float_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= 5.0
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= 5.0
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor *= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Mul_double_tensor_bool_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= self.np_bool_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor *= int_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Mul_double_tensor_int_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= self.np_int_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= self.dc_int_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor *= float_tensor
	# def test_Assignment_Mul_double_tensor_float_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= self.np_float_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= self.dc_float_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# double_tensor *= double_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_Mul_double_tensor_double_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np *= self.np_double_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc *= self.dc_double_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	


	# Assignment True_Div
	'''
	Bool tensor with Assignment True_Div operator works like Assignment Xor operator
	Numpy shows this: "numpy boolean division, the `/` operator, is deprecated, use 
				   the bitwise_xor, the `^` operator, or the logical_xor function instead."
	So skiping Bool tensor in Assignment True_Div operator
	'''
	'''
	Assignment true division divides the int tensor with another scalar or tensor, which 
	may cauuse the source tensor to change it's type to float from int. Numpy doesn't allow
	to change source tensor dtype, but Dnnc permits us to do that.
	Link to follow: (https://stackoverflow.com/questions/48948308/cant-use-on-numpy-array)

	# int_tensor /= bool_scalar
	def test_Assignment_True_Div_int_tensor_int_scalar (self):
		temp_np = self.np_int_0_4.copy()
		temp_np /= True
		temp_dc = self.dc_int_0_4.copy()
		temp_dc /= True
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor /= int_scalar
	def test_Assignment_True_Div_int_tensor_int_scalar (self):
		temp_np = self.np_int_0_4.copy()
		temp_np /= 5
		temp_dc = self.dc_int_0_4.copy()
		temp_dc /= 5
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor /= bool_tensor  
	# def test_Assignment_True_Div_int_tensor_bool_tensor (self):
	# 	temp_np = self.np_int_0_4.copy()
	# 	temp_np /= self.np_bool_5_9
	# 	temp_dc = self.dc_int_0_4.copy()
	# 	temp_dc /= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# int_tensor /= int_tensor
	def test_Assignment_True_Div_int_tensor_int_tensor (self):
		temp_np = self.np_int_0_4.copy()
		temp_np /= self.np_int_5_9
		temp_dc = self.dc_int_0_4.copy()
		temp_dc /= self.dc_int_5_9
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	'''

	# float_tensor /= bool_scalar
	def test_Assignment_True_Div_float_tensor_bool_scalar (self):
		temp_np = self.np_float_0_4.copy()
		temp_np /= True
		temp_dc = self.dc_float_0_4.copy()
		temp_dc /= True
		np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor /= int_scalar (Numpy and Dnnc has same output, but says not equal)
	# def test_Assignment_True_Div_float_tensor_int_scalar (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np /= 5
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc /= 5
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor /= float_scalar (Numpy and Dnnc has same output, but says not equal)
	# def test_Assignment_True_Div_float_tensor_float_scalar (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np /= 5.0
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc /= 5.0
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor /= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_True_Div_float_tensor_bool_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np /= self.np_bool_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc /= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor /= int_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_True_Div_float_tensor_int_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np /= self.np_int_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc /= self.dc_int_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# float_tensor /= float_tensor (Numpy and Dnnc has same output, but says not equal)
	# def test_Assignment_True_Div_float_tensor_float_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np /= self.np_float_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc /= self.dc_float_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# float_tensor /= double_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_True_Div_float_tensor_double_tensor (self):
	# 	temp_np = self.np_float_0_4.copy()
	# 	temp_np /= self.np_double_5_9
	# 	temp_dc = self.dc_float_0_4.copy()
	# 	temp_dc /= self.dc_double_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	'''
	Double True_Div tensor operator fails on Dnnc
		x: array([ 5.,  7.,  9., 11., 13.])
		y: array(<Swig Object of type 'std::vector< double,std::allocator< double > > *' at 0x7f1e95c75f80>,dtype=object)
		
		test_Assignment_True_Div_double_tensor_double_tensor (__main__.tensorDetailedOperatorsTest) ... FAIL
		swig/python detected a memory leak of type 'std::vector< double,std::allocator< double > > *', no destructor found.

	'''
	# double_tensor /= bool_scalar
	# def test_Assignment_True_Div_double_tensor_bool_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np /= True
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc /= True
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor /= int_scalar
	# def test_Assignment_True_Div_double_tensor_int_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np /= 5
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc /= 5
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor /= float_scalar
	# def test_Assignment_True_Div_double_tensor_float_scalar (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np /= 5.0
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc /= 5.0
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor /= bool_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_True_Div_double_tensor_bool_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np /= self.np_bool_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc /= self.dc_bool_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor /= int_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_True_Div_double_tensor_int_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np /= self.np_int_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc /= self.dc_int_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))

	# double_tensor /= float_tensor
	# def test_Assignment_True_Div_double_tensor_float_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np /= self.np_float_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc /= self.dc_float_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	
	# double_tensor /= double_tensor  (Numpy works, Dnnc doesn't)
	# def test_Assignment_True_Div_double_tensor_double_tensor (self):
	# 	temp_np = self.np_double_0_4.copy()
	# 	temp_np /= self.np_double_5_9
	# 	temp_dc = self.dc_double_0_4.copy()
	# 	temp_dc /= self.dc_double_5_9
	# 	np.testing.assert_array_equal(temp_np, np.array(temp_dc.data()))
	




	def tearDown(self):
		return "test finished"

if __name__ == '__main__':
	
	unittest.main()

	# array1 = np.arange(5).astype(np.int)
	# array = np.arange(5).astype(np.double)
	# print(array1)
	# array1 *=5.0
	# print(array1)

	# dc_bool_0_4 = dc.arange(5).asTypeBool()
	# temp_dc = dc_bool_0_4.copy()
	# temp_dc1 = dc_bool_0_4.copy()
	# print(temp_dc)
	# temp_dc -= True
	# print(temp_dc)
	# print(temp_dc1)
	# temp_dc1 ^= True
	# print(temp_dc1)
	
	# dc_float_0_4 = dc.arange(5).asTypeFloat()
	# dc_float_5_9 = dc.arange(5,10).asTypeFloat()
	# dc_int_0_4 = dc.arange(5).asTypeInt()
	# dc_int_5_9 = dc.arange(5,10).asTypeInt()
	# print(dc_float_0_4)
	# print(dc_float_5_9)
	# print(dc_int_0_4)
	# print(dc_int_5_9)