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

import deepC.dnnc as dc
import numpy as np
import unittest
import dnnc_testing

class tensorOperatorsTest(unittest.TestCase):
    def setUp(self):
        self.nullT=dc.array(0)
        self.zeros=dc.zeros(2,3).asTypeInt()
        self.ones=dc.ones(2,3).asTypeInt()
        self.f0_4 = dc.arange(5)
        self.f5_9 = dc.arange(10,5)
        self.np_f0_4 = np.arange(5)
        self.np_f5_9 = np.arange(10,5)
        self.i0_4 = self.f0_4.asTypeInt()
        self.i5_9 = self.f5_9.asTypeInt()
        self.np_i0_4 = self.np_f0_4.astype(np.int)
        self.np_i5_9 = self.np_f5_9.astype(np.int)
        self.b0_4 = self.f0_4.asTypeBool()
        self.b5_9 = self.f5_9.asTypeBool()

    # assignment operators
    def test_assignments(self):
        assert not self.nullT
        assert self.zeros
        assert self.ones

        # Add
        temp_zeros = self.zeros.copy()
        temp_zeros += self.ones
        dnnc_testing.utils.assert_equal( temp_zeros , self.ones )
        temp_zeros = self.zeros.copy()
        temp_zeros += 1
        dnnc_testing.utils.assert_equal( temp_zeros , self.ones )

        # Sub
        temp_ones = self.ones.copy()
        temp_ones -= self.ones
        dnnc_testing.utils.assert_equal( temp_ones , self.zeros )
        temp = self.f5_9
        temp -= dc.array([5])
        dnnc_testing.utils.assert_allclose( temp , self.f0_4 )
        temp = self.i5_9
        temp -= dc.array([5]).asTypeInt()
        dnnc_testing.utils.assert_equal( temp , self.i0_4 )
        temp = self.b5_9
        temp -= dc.array([5]).asTypeBool()
        dnnc_testing.utils.assert_equal(temp, dc.zeros(5).asTypeBool())

        # Mul
        temp_zeros = self.zeros.copy()
        temp_zeros *= self.ones
        dnnc_testing.utils.assert_equal( temp_zeros , self.zeros )
        temp_ones = self.ones.copy()
        temp_ones *= 0
        dnnc_testing.utils.assert_equal( temp_ones , self.zeros )

        # TrueDiv
        temp_zeros = self.zeros.copy().asTypeFloat()
        temp_zeros /= self.ones.asTypeFloat()
        dnnc_testing.utils.assert_equal( temp_zeros.asTypeFloat() , self.zeros.asTypeFloat() )
        temp_zeros = self.zeros.copy()
        temp_zeros /= 1
        dnnc_testing.utils.assert_equal( temp_zeros.asTypeFloat() , self.zeros.asTypeFloat() )

        # FloorDiv
        temp_zeros = self.zeros.copy()
        temp_zeros //= self.ones
        dnnc_testing.utils.assert_equal( temp_zeros , self.zeros )
        temp_zeros = self.zeros.copy()
        temp_zeros //= 1
        dnnc_testing.utils.assert_equal( temp_zeros , self.zeros )

        # Pow
        temp_zeros = self.zeros.copy()
        temp_zeros **= self.ones
        dnnc_testing.utils.assert_equal( temp_zeros , self.zeros )
        temp_ones = self.ones.copy()
        temp_ones **= 0
        dnnc_testing.utils.assert_equal( temp_ones , self.ones )

        # Mod
        temp_zeros = self.zeros.copy()
        temp_zeros %= self.ones
        dnnc_testing.utils.assert_equal( temp_zeros , self.zeros )
        temp_zeros = self.zeros.copy()
        temp_zeros %= 1
        dnnc_testing.utils.assert_equal( temp_zeros , self.zeros )

        # Left Shift
        temp_zeros = self.zeros.copy()
        temp_zeros <<= self.ones
        dnnc_testing.utils.assert_equal( temp_zeros , self.zeros << self.ones )
        temp_ones = self.ones.copy()
        temp_ones <<= 0
        dnnc_testing.utils.assert_equal( temp_ones , self.ones << 0 )

        # Right Shift
        temp_zeros = self.zeros.copy()
        temp_zeros <<= self.ones
        dnnc_testing.utils.assert_equal( temp_zeros , self.zeros >> self.ones)
        temp_ones = self.ones.copy()
        temp_ones <<= 0
        dnnc_testing.utils.assert_equal( temp_ones , self.ones >> 0 )

        # And
        temp_zeros = self.zeros.copy()
        temp_zeros &= self.ones
        dnnc_testing.utils.assert_equal( temp_zeros , self.zeros )
        temp_ones = self.ones.copy()
        temp_ones &= 0
        dnnc_testing.utils.assert_equal( temp_ones , self.zeros )

        # Or
        temp_zeros = self.zeros.copy()
        temp_zeros |= self.ones
        dnnc_testing.utils.assert_equal( temp_zeros , self.ones )
        temp_ones = self.ones.copy()
        temp_ones |= 0
        dnnc_testing.utils.assert_equal( temp_ones , self.ones )

        # Xor
        temp_zeros = self.zeros.copy()
        temp_zeros ^= self.ones
        dnnc_testing.utils.assert_equal( temp_zeros , self.ones )
        temp_ones = self.ones.copy()
        temp_ones ^= 1
        dnnc_testing.utils.assert_equal( temp_ones , self.zeros )


    # binary operators
    def test_binary(self):

        # Add
        dnnc_testing.utils.assert_equal( self.ones , self.zeros+self.ones)
        dnnc_testing.utils.assert_equal( self.ones , self.zeros+1)
        dnnc_testing.utils.assert_equal( self.ones , 1+self.zeros)
        dnnc_testing.utils.assert_equal( self.zeros+self.ones, self.ones)

        # Sub
        dnnc_testing.utils.assert_equal( -self.ones, self.zeros-self.ones)
        # dnnc_testing.utils.assert_equal( self.ones , 1-self.zeros)            # Not working for some wierd reason
        dnnc_testing.utils.assert_equal( self.ones, self.ones-self.zeros)

        # Mul
        dnnc_testing.utils.assert_equal( self.ones, self.ones*1)
        dnnc_testing.utils.assert_equal( self.ones, self.ones*self.ones)
        dnnc_testing.utils.assert_equal( self.zeros , 1*self.zeros)
        dnnc_testing.utils.assert_equal( self.zeros, self.zeros*1)
        dnnc_testing.utils.assert_equal( self.zeros, self.ones*0)

        # Left Shift
        dnnc_testing.utils.assert_equal( self.ones<<1, self.ones<<1)
        dnnc_testing.utils.assert_equal( self.ones<<self.ones, self.ones<<self.ones)
        dnnc_testing.utils.assert_equal( self.zeros<<1, self.zeros<<1)
        dnnc_testing.utils.assert_equal( self.ones<<0, self.ones<<0)

        # Right Shift
        dnnc_testing.utils.assert_equal( self.ones>>1, self.ones>>1)
        dnnc_testing.utils.assert_equal( self.ones>>self.ones, self.ones>>self.ones)
        dnnc_testing.utils.assert_equal( self.zeros>>1, self.zeros>>1)
        dnnc_testing.utils.assert_equal( self.ones>>0, self.ones>>0)

        # FloorDiv
        dnnc_testing.utils.assert_equal( self.zeros, self.zeros//self.ones)
        dnnc_testing.utils.assert_equal( self.ones, 1.0//self.ones)
        dnnc_testing.utils.assert_equal( self.zeros, self.zeros//1)
        dnnc_testing.utils.assert_equal( self.ones, self.ones//1)

        # True Div
        dnnc_testing.utils.assert_equal( self.ones.asTypeFloat(), self.ones/self.ones)
        dnnc_testing.utils.assert_equal( self.ones.asTypeFloat(), dc.true_div(self.ones, self.ones))
        dnnc_testing.utils.assert_equal( self.ones, 1.0/self.ones)
        dnnc_testing.utils.assert_equal( self.zeros.asTypeFloat(), self.zeros/1.0)
        dnnc_testing.utils.assert_equal( self.ones.asTypeFloat(), self.ones/1.0)

        # Pow
        dnnc_testing.utils.assert_equal( self.ones , self.ones**self.ones)
        dnnc_testing.utils.assert_equal( self.ones , self.ones**1)
        dnnc_testing.utils.assert_equal( self.ones , self.ones**0)

        # And
        dnnc_testing.utils.assert_equal( self.zeros.asTypeBool() , self.zeros&self.ones)
        dnnc_testing.utils.assert_equal( self.ones , 1&self.ones) 
        dnnc_testing.utils.assert_equal( self.zeros , 0&self.ones)
        dnnc_testing.utils.assert_equal( self.ones.asTypeBool() , self.ones&1)
        dnnc_testing.utils.assert_equal( self.zeros.asTypeBool() , self.ones&0)

        # Or
        dnnc_testing.utils.assert_equal( self.ones.asTypeBool() , self.zeros|self.ones)
        dnnc_testing.utils.assert_equal( self.ones , 0|self.ones) 
        dnnc_testing.utils.assert_equal( self.ones , 1|self.zeros)
        dnnc_testing.utils.assert_equal( self.ones.asTypeBool() , self.ones|0)
        dnnc_testing.utils.assert_equal( self.ones.asTypeBool() , self.zeros|1)

        # Xor
        dnnc_testing.utils.assert_equal( self.ones.asTypeBool() , self.zeros^self.ones)
        dnnc_testing.utils.assert_equal( self.ones , 0^self.ones) 
        dnnc_testing.utils.assert_equal( self.zeros , 0^self.zeros)
        dnnc_testing.utils.assert_equal( self.ones.asTypeBool() , self.ones^0)
        dnnc_testing.utils.assert_equal( self.zeros.asTypeBool() , self.zeros^0)


    # unary operators
    def test_unary(self):

        # Neg
        dnnc_testing.utils.assert_equal( self.i0_4, -(-self.i0_4))

        # Pos
        dnnc_testing.utils.assert_equal( -self.i0_4, +(-self.i0_4))

        # Abs
        dnnc_testing.utils.assert_equal( self.i0_4, abs(-self.i0_4))

        # Invert
        dnnc_testing.utils.assert_equal( self.zeros.asTypeBool() , ~self.ones)
        dnnc_testing.utils.assert_equal( self.ones.asTypeBool() , ~self.zeros)
        dnnc_testing.utils.assert_equal( ~~~self.f5_9 , ~self.f5_9)


    # comparison operators
    def test_comparison(self):

        # Less
        lessResult = self.zeros < self.ones
        dnnc_testing.utils.assert_equal(lessResult, self.ones)

        # LessEqual
        lessEqualResult = self.zeros <= self.zeros
        dnnc_testing.utils.assert_equal(lessEqualResult, self.ones)

        # Greater
        greaterResult = self.ones > self.zeros
        dnnc_testing.utils.assert_equal(greaterResult, self.ones)

        # GreaterEqual
        greaterEqualResult = self.ones >= self.ones
        dnnc_testing.utils.assert_equal(greaterEqualResult, self.ones)

        # Equal
        equalResult = self.ones == dc.ones(2,3).asTypeInt()
        dnnc_testing.utils.assert_equal(equalResult, self.ones)

        # NotEqual
        notEqualResult = self.ones != self.zeros
        dnnc_testing.utils.assert_equal(notEqualResult, self.ones)


    def tearDown(self):
        return "test finished"


if __name__ == '__main__':
    unittest.main()

