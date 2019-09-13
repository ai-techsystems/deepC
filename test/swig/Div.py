# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for divitional information
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

import common

import dnnc as dc
import numpy as np
import unittest

class DivTest(unittest.TestCase):
   
    def setUp(self):
        self.len = 48

        self.np_int_a = np.random.randn(self.len).astype(np.int)
        self.np_int_b = np.random.randn(self.len).astype(np.int)
        self.dc_int_a = dc.array(list(self.np_int_a))
        self.dc_int_b = dc.array(list(self.np_int_b))

        self.np_float_a = np.random.randn(self.len).astype(np.float32)
        self.np_float_b = np.random.randn(self.len).astype(np.float32)
        self.dc_float_a = dc.array(list(self.np_float_a))
        self.dc_float_b = dc.array(list(self.np_float_b))

        self.np_double_a = np.random.randn(self.len).astype(np.float64)
        self.np_double_b = np.random.randn(self.len).astype(np.float64)
        self.dc_double_a = dc.array(list(self.np_double_a))
        self.dc_double_b = dc.array(list(self.np_double_b))


    # def test_Div1D_int (self):
    #     npr = np.divide(self.np_int_a, self.np_int_b)
    #     dcr = dc.div(self.dc_int_a, self.dc_int_b)
    #     np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.int),
    #             rtol=1e-3, atol=1e-3)

    def test_Div1D_float (self):
        npr = np.divide(self.np_float_a, self.np_float_b)
        dcr = dc.div(self.dc_float_a, self.dc_float_b)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Div1D_double (self):
        npr = np.divide(self.np_double_a, self.np_double_b)
        dcr = dc.div(self.dc_double_a, self.dc_double_b)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    # def test_Div2D_int (self):
    #     np_int_a = np.reshape(self.np_int_a, (6,8))
    #     np_int_b = np.reshape(self.np_int_b, (6,8))
    #     dc_int_a = dc.reshape(self.dc_int_a, (6,8))
    #     dc_int_b = dc.reshape(self.dc_int_b, (6,8))
    #     npr = np.divide(np_int_a, np_int_b)
    #     dcr = dc.div(dc_int_a, dc_int_b)
    #     np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
    #             rtol=1e-3, atol=1e-3)

    def test_Div2D_float (self):
        np_float_a = np.reshape(self.np_float_a, (6,8))
        np_float_b = np.reshape(self.np_float_b, (6,8))
        dc_float_a = dc.reshape(self.dc_float_a, (6,8))
        dc_float_b = dc.reshape(self.dc_float_b, (6,8))
        npr = np.divide(np_float_a, np_float_b)
        dcr = dc.div(dc_float_a, dc_float_b)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Div2D_double (self):
        np_double_a = np.reshape(self.np_double_a, (6,8))
        np_double_b = np.reshape(self.np_double_b, (6,8))
        dc_double_a = dc.reshape(self.dc_double_a, (6,8))
        dc_double_b = dc.reshape(self.dc_double_b, (6,8))
        npr = np.divide(np_double_a, np_double_b)
        dcr = dc.div(dc_double_a, dc_double_b)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    # def test_Div3D_int (self):
    #     np_int_a = np.reshape(self.np_int_a, (4,4,3))
    #     np_int_b = np.reshape(self.np_int_b, (4,4,3))
    #     dc_int_a = dc.reshape(self.dc_int_a, (4,4,3))
    #     dc_int_b = dc.reshape(self.dc_int_b, (4,4,3))
    #     npr = np.divide(np_int_a, np_int_b)
    #     dcr = dc.div(dc_int_a, dc_int_b)
    #     np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
    #             rtol=1e-3, atol=1e-3)

    def test_Div3D_float (self):
        np_float_a = np.reshape(self.np_float_a, (4,4,3))
        np_float_b = np.reshape(self.np_float_b, (4,4,3))
        dc_float_a = dc.reshape(self.dc_float_a, (4,4,3))
        dc_float_b = dc.reshape(self.dc_float_b, (4,4,3))
        npr = np.divide(np_float_a, np_float_b)
        dcr = dc.div(dc_float_a, dc_float_b)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Div3D_double (self):
        np_double_a = np.reshape(self.np_double_a, (4,4,3))
        np_double_b = np.reshape(self.np_double_b, (4,4,3))
        dc_double_a = dc.reshape(self.dc_double_a, (4,4,3))
        dc_double_b = dc.reshape(self.dc_double_b, (4,4,3))
        npr = np.divide(np_double_a, np_double_b)
        dcr = dc.div(dc_double_a, dc_double_b)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    # def test_Div4D_int (self):
    #     np_int_a = np.reshape(self.np_int_a, (4,2,2,3))
    #     np_int_b = np.reshape(self.np_int_b, (4,2,2,3))
    #     dc_int_a = dc.reshape(self.dc_int_a, (4,2,2,3))
    #     dc_int_b = dc.reshape(self.dc_int_b, (4,2,2,3))
    #     npr = np.divide(np_int_a, np_int_b)
    #     dcr = dc.div(dc_int_a, dc_int_b)
    #     np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
    #             rtol=1e-3, atol=1e-3)

    def test_Div4D_float (self):
        np_float_a = np.reshape(self.np_float_a, (4,2,2,3))
        np_float_b = np.reshape(self.np_float_b, (4,2,2,3))
        dc_float_a = dc.reshape(self.dc_float_a, (4,2,2,3))
        dc_float_b = dc.reshape(self.dc_float_b, (4,2,2,3))
        npr = np.divide(np_float_a, np_float_b)
        dcr = dc.div(dc_float_a, dc_float_b)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Div4D_double (self):
        np_double_a = np.reshape(self.np_double_a, (4,2,2,3))
        np_double_b = np.reshape(self.np_double_b, (4,2,2,3))
        dc_double_a = dc.reshape(self.dc_double_a, (4,2,2,3))
        dc_double_b = dc.reshape(self.dc_double_b, (4,2,2,3))
        npr = np.divide(np_double_a, np_double_b)
        dcr = dc.div(dc_double_a, dc_double_b)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
    
