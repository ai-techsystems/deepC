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
import dnnc_testing

class tensorOperatorsTest(unittest.TestCase):
    def setUp(self):
        self.nullT=dc.array(0)
        self.zeros=dc.zeros(2,3).asTypeInt()
        self.ones=dc.ones(2,3).asTypeInt()
        self.f0_4 = dc.arange(5);
        self.f5_9 = dc.arange(10,5);
        self.np_f0_4 = np.arange(5);
        self.np_f5_9 = np.arange(10,5);
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
        self.zeros += self.ones
        dnnc_testing.utils.assert_equal( self.zeros , self.ones )
        self.zeros -= self.ones
        tmp = self.ones.copy();
        tmp -= self.ones;
        dnnc_testing.utils.assert_equal( tmp , self.zeros )
        tmp = self.f5_9
        tmp -= dc.array([5]);
        dnnc_testing.utils.assert_allclose( tmp , self.f0_4 )
        tmp = self.i5_9
        tmp -= dc.array([5]).asTypeInt();
        dnnc_testing.utils.assert_equal( tmp , self.i0_4 )
        tmp = self.b5_9
        tmp -= dc.array([5]).asTypeBool();

        dnnc_testing.utils.assert_equal(tmp, dc.zeros(5).asTypeBool())

    # binary operators
    def test_binary(self):
        # Add
        dnnc_testing.utils.assert_equal( self.ones , self.zeros+self.ones)
        dnnc_testing.utils.assert_equal( self.ones , self.zeros+1)
        dnnc_testing.utils.assert_equal( self.zeros+self.ones, self.ones)
        # Sub
        dnnc_testing.utils.assert_equal( -self.ones, self.zeros-self.ones)
        dnnc_testing.utils.assert_equal( self.ones, self.ones-self.zeros)
        # And
        dnnc_testing.utils.assert_equal( self.zeros , self.zeros&self.ones)
        # dnnc_testing.utils.assert_equal( self.ones , 1&self.ones)   # And doesn't work with scalar 
        # dnnc_testing.utils.assert_equal( self.zeros , 0&self.ones)  # on left and tensor on right 
        dnnc_testing.utils.assert_equal( self.ones , self.ones&1)
        dnnc_testing.utils.assert_equal( self.zeros , self.ones&0)
        # Or
        # dnnc_testing.utils.assert_equal( self.ones , self.zeros|self.ones)
        # dnnc_testing.utils.assert_equal( self.ones , 0|self.ones)   # Or doesn't work with scalar 
        # dnnc_testing.utils.assert_equal( self.ones , 1|self.zeros)  # on left and tensor on right 
        # dnnc_testing.utils.assert_equal( self.ones , self.ones|0)
        # dnnc_testing.utils.assert_equal( self.ones , self.zeros|1)
        # Pow
        # dnnc_testing.utils.assert_equal( self.ones , self.ones**1)
        # dnnc_testing.utils.assert_equal( np.power(self.np_i0_4 , self.np_i5_9), self.i0_4**self.i5_9)

    # unary operators
    def test_unary(self):
        # Neg
        dnnc_testing.utils.assert_equal( self.i0_4, -(-self.i0_4))
        # Pos
        dnnc_testing.utils.assert_equal( -self.i0_4, +(-self.i0_4))
        # Abs
        dnnc_testing.utils.assert_equal( self.i0_4, abs(-self.i0_4))

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

