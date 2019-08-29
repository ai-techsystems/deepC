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

def temp_dequantize_linear(np_x, np_x_scale, np_x_zero_point):
    y = (np_x - np_x_zero_point) * np_x_scale
    return y

class DequantizeLinearTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_x = np.random.randn(self.len).astype(np.float32)
        self.np_x_scale = np.random.randn(1).astype(np.float32)
        self.np_x_zero_point = np.random.randn(1).astype(np.float32)
        self.dc_x = dc.array(list(self.np_x))
        self.dc_x_scale = dc.array(list(self.np_x_scale))
        self.dc_x_zero_point = dc.array(list(self.np_x_zero_point))

    def test_DequantizeLinear1D (self):
        npr = temp_dequantize_linear(self.np_x, self.np_x_scale, self.np_x_zero_point)
        dcr = dc.dequantize_linear(self.dc_x, self.dc_x_scale, self.dc_x_zero_point)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_DequantizeLinear2D (self):
        np_x = np.reshape(self.np_x, (6,4))
        dc_x = dc.reshape(self.dc_x, (6,4))
        npr = temp_dequantize_linear(np_x, self.np_x_scale, self.np_x_zero_point)
        dcr = dc.dequantize_linear(dc_x, self.dc_x_scale, self.dc_x_zero_point)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_DequantizeLinear3D (self):
        np_x = np.reshape(self.np_x, (2,4,3))
        dc_x = dc.reshape(self.dc_x, (2,4,3))
        npr = temp_dequantize_linear(np_x, self.np_x_scale, self.np_x_zero_point)
        dcr = dc.dequantize_linear(dc_x, self.dc_x_scale, self.dc_x_zero_point)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_DequantizeLinear4D (self):
        np_x = np.reshape(self.np_x, (2,2,2,3))
        dc_x = dc.reshape(self.dc_x, (2,2,2,3))
        npr = temp_dequantize_linear(np_x, self.np_x_scale, self.np_x_zero_point)
        dcr = dc.dequantize_linear(dc_x, self.dc_x_scale, self.dc_x_zero_point)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
    
