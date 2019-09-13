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


class EyeLikeTest(unittest.TestCase):
    
    def setUp(self):
        self.len = 48
        self.k = np.random.randint(low=-10, high=10)

        self.np_bool_a = np.random.randn(self.len).astype(np.bool)
        self.np_bool_b = np.random.randn(self.len).astype(np.bool)
        self.dc_bool_a = dc.array(list(self.np_bool_a))
        self.dc_bool_b = dc.array(list(self.np_bool_b))

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


    # EyeLike by default takes 2D tensor only

    def test_EyeLike2D_bool_1 (self):
        np_a = np.reshape(self.np_bool_a, (12,4))
        dc_a = dc.reshape(self.dc_bool_a, (12,4))
        npr = np.eye(12, 4, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.bool),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_bool_2 (self):
        np_a = np.reshape(self.np_bool_a, (2,24))
        dc_a = dc.reshape(self.dc_bool_a, (2,24))
        npr = np.eye(2, 24, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.bool),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_bool_3 (self):
        np_a = np.reshape(self.np_bool_a, (6,8))
        dc_a = dc.reshape(self.dc_bool_a, (6,8))
        npr = np.eye(6, 8, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.bool),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_int_1 (self):
        np_a = np.reshape(self.np_int_a, (12,4))
        dc_a = dc.reshape(self.dc_int_a, (12,4))
        npr = np.eye(12, 4, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_int_2 (self):
        np_a = np.reshape(self.np_int_a, (2,24))
        dc_a = dc.reshape(self.dc_int_a, (2,24))
        npr = np.eye(2, 24, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_int_3 (self):
        np_a = np.reshape(self.np_int_a, (6,8))
        dc_a = dc.reshape(self.dc_int_a, (6,8))
        npr = np.eye(6, 8, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_float_1 (self):
        np_a = np.reshape(self.np_float_a, (12,4))
        dc_a = dc.reshape(self.dc_float_a, (12,4))
        npr = np.eye(12, 4, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_float_2 (self):
        np_a = np.reshape(self.np_float_a, (2,24))
        dc_a = dc.reshape(self.dc_float_a, (2,24))
        npr = np.eye(2, 24, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_float_3 (self):
        np_a = np.reshape(self.np_float_a, (6,8))
        dc_a = dc.reshape(self.dc_float_a, (6,8))
        npr = np.eye(6, 8, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_double_1 (self):
        np_a = np.reshape(self.np_double_a, (12,4))
        dc_a = dc.reshape(self.dc_double_a, (12,4))
        npr = np.eye(12, 4, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_double_2 (self):
        np_a = np.reshape(self.np_double_a, (2,24))
        dc_a = dc.reshape(self.dc_double_a, (2,24))
        npr = np.eye(2, 24, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def test_EyeLike2D_double_3 (self):
        np_a = np.reshape(self.np_double_a, (6,8))
        dc_a = dc.reshape(self.dc_double_a, (6,8))
        npr = np.eye(6, 8, k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
    
