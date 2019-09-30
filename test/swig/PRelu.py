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

import dnnc as dc
import numpy as np
import unittest

def prelu_util(x, slope):
    arr = x.copy()
    l =  len(slope)
    for i, e in enumerate(arr):
        if e < 0:
            if l == 1:
                arr[i] = e * slope[0]
            else: 
                arr[i] = e * slope[i]
    return arr

class PReluTest(unittest.TestCase):
    def setUp(self):
        self.len = 24

        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a))

        self.np_slope_1 = np.random.randn(1).astype(np.float32)
        self.dc_slope_1 = dc.array(list(self.np_slope_1))

        self.np_slope = np.random.randn(self.len).astype(np.float32)
        self.dc_slope = dc.array(list(self.np_slope))

        self.prelu_true = prelu_util(self.np_a, self.np_slope)
        self.prelu_true_1 = prelu_util(self.np_a, self.np_slope_1)


    def test_prelu_1d (self):
        dcr = dc.prelu(self.dc_a, self.dc_slope)

        np.testing.assert_allclose(self.prelu_true, np.array(dcr.data()).astype(np.float32),
            rtol=1e-3, atol=1e-3)

    def test_prelu_1d_broadcast (self):
        dcr = dc.prelu(self.dc_a,self.dc_slope_1)
        
        np.testing.assert_allclose(self.prelu_true_1, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_prelu_2d (self):
        dc_a_reshaped = dc.reshape(self.dc_a, (6, 4))
        dc_slope_reshaped = dc.reshape(self.dc_slope, (6, 4))

        np_test = np.reshape(self.prelu_true, (6, 4))

        dc_test = dc.prelu(dc_a_reshaped, dc_slope_reshaped)
        np.testing.assert_allclose(np_test.flatten(), np.array(dc_test.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_prelu_2d_broadcast (self):
        dc_a_reshaped = dc.reshape(self.dc_a, (6, 4))

        np_test = self.prelu_true_1.copy()

        dc_test = dc.prelu(dc_a_reshaped, self.dc_slope_1)
        np.testing.assert_allclose(np_test.flatten(), np.array(dc_test.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_prelu_3d (self):
        dc_a_reshaped = dc.reshape(self.dc_a, (2, 4, 3))
        dc_slope_reshaped = dc.reshape(self.dc_slope, (2, 4, 3))

        np_test = np.reshape(self.prelu_true, (2, 4, 3))

        dc_test = dc.prelu(dc_a_reshaped, dc_slope_reshaped)
        np.testing.assert_allclose(np_test.flatten(), np.array(dc_test.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_prelu_3d_broadcast (self):
        dc_a_reshaped = dc.reshape(self.dc_a, (2, 4, 3))

        np_test = self.prelu_true_1.copy()

        dc_test = dc.prelu(dc_a_reshaped, self.dc_slope_1)
        np.testing.assert_allclose(np_test.flatten(), np.array(dc_test.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_prelu_4d (self):
        dc_a_reshaped = dc.reshape(self.dc_a, (2, 3, 2, 2))
        dc_slope_reshaped = dc.reshape(self.dc_slope, (2, 3, 2, 2))

        np_test = np.reshape(self.prelu_true, (2, 3, 2, 2))

        dc_test = dc.prelu(dc_a_reshaped, dc_slope_reshaped)
        np.testing.assert_allclose(np_test.flatten(), np.array(dc_test.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_prelu_4d_broadcast (self):
        dc_a_reshaped = dc.reshape(self.dc_a, (2, 1, 4, 3))

        np_test = self.prelu_true_1.copy()

        dc_test = dc.prelu(dc_a_reshaped, self.dc_slope_1)
        np.testing.assert_allclose(np_test.flatten(), np.array(dc_test.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
