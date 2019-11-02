# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
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

def temp_flatten(x, shape, axis):
    new_shape = (1, -1) if axis == 0 else (np.prod(shape[0:axis]).astype(int), -1)
    y = np.reshape(x, new_shape)
    return y

class FlattenTest(unittest.TestCase):
    def setUp(self):
        self.len = 48

        self.np_bool_a = np.random.randn(self.len).astype(np.bool)
        self.dc_bool_a = dc.array(list(self.np_bool_a))
        
        self.np_int_a = np.random.randn(self.len).astype(np.int)
        self.dc_int_a = dc.array(list(self.np_int_a))
        
        self.np_float_a = np.random.randn(self.len).astype(np.float32)
        self.dc_float_a = dc.array(list(self.np_float_a))

        self.np_double_a = np.random.randn(self.len).astype(np.float64)
        self.dc_double_a = dc.array(list(self.np_double_a))

    def test_Flatten1D_bool (self):
        axis = 0
        shape = (1,48)
        npr = temp_flatten(self.np_bool_a, shape, axis)
        dcr = dc.flatten(self.dc_bool_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.bool),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten1D_int (self):
        axis = 0
        shape = (1,48)
        npr = temp_flatten(self.np_int_a, shape, axis)
        dcr = dc.flatten(self.dc_int_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten1D_float (self):
        axis = 0
        shape = (1,48)
        npr = temp_flatten(self.np_float_a, shape, axis)
        dcr = dc.flatten(self.dc_float_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten1D_double (self):
        axis = 0
        shape = (1,48)
        npr = temp_flatten(self.np_double_a, shape, axis)
        dcr = dc.flatten(self.dc_double_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())


    def test_Flatten2D_bool (self):
        axis = 2
        shape = (8,6)
        np_bool_a = np.reshape(self.np_bool_a, shape)
        dc_bool_a = dc.reshape(self.dc_bool_a, shape)
        npr = temp_flatten(np_bool_a, shape, axis)
        dcr = dc.flatten(dc_bool_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.bool),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten2D_int (self):
        axis = 2
        shape = (8,6)
        np_int_a = np.reshape(self.np_int_a, shape)
        dc_int_a = dc.reshape(self.dc_int_a, shape)
        npr = temp_flatten(np_int_a, shape, axis)
        dcr = dc.flatten(dc_int_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten2D_float (self):
        axis = 2
        shape = (8,6)
        np_float_a = np.reshape(self.np_float_a, shape)
        dc_float_a = dc.reshape(self.dc_float_a, shape)
        npr = temp_flatten(np_float_a, shape, axis)
        dcr = dc.flatten(dc_float_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten2D_double (self):
        axis = 2
        shape = (8,6)
        np_double_a = np.reshape(self.np_double_a, shape)
        dc_double_a = dc.reshape(self.dc_double_a, shape)
        npr = temp_flatten(np_double_a, shape, axis)
        dcr = dc.flatten(dc_double_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())


    def test_Flatten3D_bool (self):
        axis = 2
        shape = (4,4,3)
        np_bool_a = np.reshape(self.np_bool_a, shape)
        dc_bool_a = dc.reshape(self.dc_bool_a, shape)
        npr = temp_flatten(np_bool_a, shape, axis)
        dcr = dc.flatten(dc_bool_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.bool),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten3D_int (self):
        axis = 2
        shape = (4,4,3)
        np_int_a = np.reshape(self.np_int_a, shape)
        dc_int_a = dc.reshape(self.dc_int_a, shape)
        npr = temp_flatten(np_int_a, shape, axis)
        dcr = dc.flatten(dc_int_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten3D_float (self):
        axis = 2
        shape = (4,4,3)
        np_float_a = np.reshape(self.np_float_a, shape)
        dc_float_a = dc.reshape(self.dc_float_a, shape)
        npr = temp_flatten(np_float_a, shape, axis)
        dcr = dc.flatten(dc_float_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten3D_double (self):
        axis = 2
        shape = (4,4,3)
        np_double_a = np.reshape(self.np_double_a, shape)
        dc_double_a = dc.reshape(self.dc_double_a, shape)
        npr = temp_flatten(np_double_a, shape, axis)
        dcr = dc.flatten(dc_double_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())


    def test_Flatten4D_bool (self):
        axis = 3
        shape = (4,2,2,3)
        np_bool_a = np.reshape(self.np_bool_a, shape)
        dc_bool_a = dc.reshape(self.dc_bool_a, shape)
        npr = temp_flatten(np_bool_a, shape, axis)
        dcr = dc.flatten(dc_bool_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.bool),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten4D_int (self):
        axis = 3
        shape = (4,2,2,3)
        np_int_a = np.reshape(self.np_int_a, shape)
        dc_int_a = dc.reshape(self.dc_int_a, shape)
        npr = temp_flatten(np_int_a, shape, axis)
        dcr = dc.flatten(dc_int_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten4D_float (self):
        axis = 3
        shape = (4,2,2,3)
        np_float_a = np.reshape(self.np_float_a, shape)
        dc_float_a = dc.reshape(self.dc_float_a, shape)
        npr = temp_flatten(np_float_a, shape, axis)
        dcr = dc.flatten(dc_float_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def test_Flatten4D_double (self):
        axis = 3
        shape = (4,2,2,3)
        np_double_a = np.reshape(self.np_double_a, shape)
        dc_double_a = dc.reshape(self.dc_double_a, shape)
        npr = temp_flatten(np_double_a, shape, axis)
        dcr = dc.flatten(dc_double_a, axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)
        np.testing.assert_equal(npr.shape, dcr.shape())

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
