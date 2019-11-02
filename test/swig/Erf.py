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

import math
import deepC.dnnc as dc
import numpy as np
import unittest

def temp_erf(x):
    y = np.vectorize(math.erf)(x).astype(np.float32)
    return y

class ErfTest(unittest.TestCase):
    def setUp(self):
        self.len = 48

        self.np_float_a = np.random.randn(self.len).astype(np.float32)
        self.dc_float_a = dc.array(list(self.np_float_a))

        self.np_double_a = np.random.randn(self.len).astype(np.float64)
        self.dc_double_a = dc.array(list(self.np_double_a))

    def test_Erf1D_float (self):
        npr = temp_erf(self.np_float_a)
        dcr = dc.erf(self.dc_float_a)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Erf1D_double (self):
        npr = temp_erf(self.np_double_a)
        dcr = dc.erf(self.dc_double_a)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def test_Erf2D_float_1 (self):
        np_float_a = np.reshape(self.np_float_a, (3,16))
        dc_float_a = dc.reshape(self.dc_float_a, (3,16))
        npr = temp_erf(np_float_a)
        dcr = dc.erf(dc_float_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Erf2D_float_2 (self):
        np_float_a = np.reshape(self.np_float_a, (6,8))
        dc_float_a = dc.reshape(self.dc_float_a, (6,8))
        npr = temp_erf(np_float_a)
        dcr = dc.erf(dc_float_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Erf2D_float_3 (self):
        np_float_a = np.reshape(self.np_float_a, (12,4))
        dc_float_a = dc.reshape(self.dc_float_a, (12,4))
        npr = temp_erf(np_float_a)
        dcr = dc.erf(dc_float_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Erf2D_double_1 (self):
        np_double_a = np.reshape(self.np_double_a, (3,16))
        dc_double_a = dc.reshape(self.dc_double_a, (3,16))
        npr = temp_erf(np_double_a)
        dcr = dc.erf(dc_double_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def test_Erf2D_double_2 (self):
        np_double_a = np.reshape(self.np_double_a, (6,8))
        dc_double_a = dc.reshape(self.dc_double_a, (6,8))
        npr = temp_erf(np_double_a)
        dcr = dc.erf(dc_double_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def test_Erf2D_double_3 (self):
        np_double_a = np.reshape(self.np_double_a, (12,4))
        dc_double_a = dc.reshape(self.dc_double_a, (12,4))
        npr = temp_erf(np_double_a)
        dcr = dc.erf(dc_double_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def test_Erf3D_float_1 (self):
        np_float_a = np.reshape(self.np_float_a, (4,4,3))
        dc_float_a = dc.reshape(self.dc_float_a, (4,4,3))
        npr = temp_erf(np_float_a)
        dcr = dc.erf(dc_float_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Erf3D_float_2 (self):
        np_float_a = np.reshape(self.np_float_a, (8,2,3))
        dc_float_a = dc.reshape(self.dc_float_a, (8,2,3))
        npr = temp_erf(np_float_a)
        dcr = dc.erf(dc_float_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Erf3D_float_3 (self):
        np_float_a = np.reshape(self.np_float_a, (2,4,6))
        dc_float_a = dc.reshape(self.dc_float_a, (2,4,6))
        npr = temp_erf(np_float_a)
        dcr = dc.erf(dc_float_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Erf3D_double_1 (self):
        np_double_a = np.reshape(self.np_double_a, (4,4,3))
        dc_double_a = dc.reshape(self.dc_double_a, (4,4,3))
        npr = temp_erf(np_double_a)
        dcr = dc.erf(dc_double_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def test_Erf3D_double_2 (self):
        np_double_a = np.reshape(self.np_double_a, (8,2,3))
        dc_double_a = dc.reshape(self.dc_double_a, (8,2,3))
        npr = temp_erf(np_double_a)
        dcr = dc.erf(dc_double_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def test_Erf3D_double_3 (self):
        np_double_a = np.reshape(self.np_double_a, (2,4,6))
        dc_double_a = dc.reshape(self.dc_double_a, (2,4,6))
        npr = temp_erf(np_double_a)
        dcr = dc.erf(dc_double_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def test_Erf4D_float (self):
        np_float_a = np.reshape(self.np_float_a, (4,2,2,3))
        dc_float_a = dc.reshape(self.dc_float_a, (4,2,2,3))
        npr = temp_erf(np_float_a)
        dcr = dc.erf(dc_float_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Erf4D_double (self):
        np_double_a = np.reshape(self.np_double_a, (4,2,2,3))
        dc_double_a = dc.reshape(self.dc_double_a, (4,2,2,3))
        npr = temp_erf(np_double_a)
        dcr = dc.erf(dc_double_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()