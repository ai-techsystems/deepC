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

def temp_gemm(np_a, np_b, np_c, alpha, beta, transA, transB):
    np_a = np_a.T if (transA==1) else np_a
    np_b = np_b.T if (transB==1) else np_b
    y = (alpha * np.dot(np_a, np_b)) + (beta * np_c)
    return y

class GemmTest(unittest.TestCase):
    def setUp(self):
        self.len_a_b = 48
        self.len_c = 64
        self.alpha = 0.5
        self.beta = 0.5

        self.np_float_a = np.random.randn(self.len_a_b).astype(np.float32)
        self.np_float_b = np.random.randn(self.len_a_b).astype(np.float32)
        self.np_float_c = np.random.randn(self.len_c).astype(np.float32)
        self.dc_float_a = dc.array(list(self.np_float_a))
        self.dc_float_b = dc.array(list(self.np_float_b))
        self.dc_float_c = dc.array(list(self.np_float_c))
        
        self.np_double_a = np.random.randn(self.len_a_b).astype(np.double)
        self.np_double_b = np.random.randn(self.len_a_b).astype(np.double)
        self.np_double_c = np.random.randn(self.len_c).astype(np.double)
        self.dc_double_a = dc.array(list(self.np_double_a))
        self.dc_double_b = dc.array(list(self.np_double_b))
        self.dc_double_c = dc.array(list(self.np_double_c))

    # Gemm by default takes 2D tensor only

    def test_Gemm2D_float_1 (self):
        shape_a = (8,6) 
        shape_b = (6,8) 
        shape_c = (8,8) 
        transA = 0
        transB = 0
        np_float_a = np.reshape(self.np_float_a, shape_a)
        np_float_b = np.reshape(self.np_float_b, shape_b)
        np_float_c = np.reshape(self.np_float_c, shape_c)
        dc_float_a = dc.reshape(self.dc_float_a, shape_a)
        dc_float_b = dc.reshape(self.dc_float_b, shape_b)
        dc_float_c = dc.reshape(self.dc_float_c, shape_c)
        npr = temp_gemm(np_float_a, np_float_b, np_float_c, self.alpha, self.beta, transA, transB)
        dcr = dc.gemm(dc_float_a, dc_float_b, dc_float_c, self.alpha, self.beta, transA, transB)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Gemm2D_float_2 (self):
        shape_a = (8,6) 
        shape_b = (8,6) 
        shape_c = (8,8) 
        transA = 0
        transB = 1
        np_float_a = np.reshape(self.np_float_a, shape_a)
        np_float_b = np.reshape(self.np_float_b, shape_b)
        np_float_c = np.reshape(self.np_float_c, shape_c)
        dc_float_a = dc.reshape(self.dc_float_a, shape_a)
        dc_float_b = dc.reshape(self.dc_float_b, shape_b)
        dc_float_c = dc.reshape(self.dc_float_c, shape_c)
        npr = temp_gemm(np_float_a, np_float_b, np_float_c, self.alpha, self.beta, transA, transB)
        dcr = dc.gemm(dc_float_a, dc_float_b, dc_float_c, self.alpha, self.beta, transA, transB)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)
        
    def test_Gemm2D_float_3 (self):
        shape_a = (6,8) 
        shape_b = (6,8) 
        shape_c = (8,8) 
        transA = 1
        transB = 0
        np_float_a = np.reshape(self.np_float_a, shape_a)
        np_float_b = np.reshape(self.np_float_b, shape_b)
        np_float_c = np.reshape(self.np_float_c, shape_c)
        dc_float_a = dc.reshape(self.dc_float_a, shape_a)
        dc_float_b = dc.reshape(self.dc_float_b, shape_b)
        dc_float_c = dc.reshape(self.dc_float_c, shape_c)
        npr = temp_gemm(np_float_a, np_float_b, np_float_c, self.alpha, self.beta, transA, transB)
        dcr = dc.gemm(dc_float_a, dc_float_b, dc_float_c, self.alpha, self.beta, transA, transB)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)
        
    def test_Gemm2D_float_4 (self):
        shape_a = (6,8) 
        shape_b = (8,6) 
        shape_c = (8,8) 
        transA = 1
        transB = 1
        np_float_a = np.reshape(self.np_float_a, shape_a)
        np_float_b = np.reshape(self.np_float_b, shape_b)
        np_float_c = np.reshape(self.np_float_c, shape_c)
        dc_float_a = dc.reshape(self.dc_float_a, shape_a)
        dc_float_b = dc.reshape(self.dc_float_b, shape_b)
        dc_float_c = dc.reshape(self.dc_float_c, shape_c)
        npr = temp_gemm(np_float_a, np_float_b, np_float_c, self.alpha, self.beta, transA, transB)
        dcr = dc.gemm(dc_float_a, dc_float_b, dc_float_c, self.alpha, self.beta, transA, transB)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)
        
    def test_Gemm2D_double_1 (self):
        shape_a = (8,6) 
        shape_b = (6,8) 
        shape_c = (8,8) 
        transA = 0
        transB = 0
        np_double_a = np.reshape(self.np_double_a, shape_a)
        np_double_b = np.reshape(self.np_double_b, shape_b)
        np_double_c = np.reshape(self.np_double_c, shape_c)
        dc_double_a = dc.reshape(self.dc_double_a, shape_a)
        dc_double_b = dc.reshape(self.dc_double_b, shape_b)
        dc_double_c = dc.reshape(self.dc_double_c, shape_c)
        npr = temp_gemm(np_double_a, np_double_b, np_double_c, self.alpha, self.beta, transA, transB)
        dcr = dc.gemm(dc_double_a, dc_double_b, dc_double_c, self.alpha, self.beta, transA, transB)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.double),
                rtol=1e-3, atol=1e-3)

    def test_Gemm2D_double_2 (self):
        shape_a = (8,6) 
        shape_b = (8,6) 
        shape_c = (8,8) 
        transA = 0
        transB = 1
        np_double_a = np.reshape(self.np_double_a, shape_a)
        np_double_b = np.reshape(self.np_double_b, shape_b)
        np_double_c = np.reshape(self.np_double_c, shape_c)
        dc_double_a = dc.reshape(self.dc_double_a, shape_a)
        dc_double_b = dc.reshape(self.dc_double_b, shape_b)
        dc_double_c = dc.reshape(self.dc_double_c, shape_c)
        npr = temp_gemm(np_double_a, np_double_b, np_double_c, self.alpha, self.beta, transA, transB)
        dcr = dc.gemm(dc_double_a, dc_double_b, dc_double_c, self.alpha, self.beta, transA, transB)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.double),
                rtol=1e-3, atol=1e-3)
        
    def test_Gemm2D_double_3 (self):
        shape_a = (6,8) 
        shape_b = (6,8) 
        shape_c = (8,8) 
        transA = 1
        transB = 0
        np_double_a = np.reshape(self.np_double_a, shape_a)
        np_double_b = np.reshape(self.np_double_b, shape_b)
        np_double_c = np.reshape(self.np_double_c, shape_c)
        dc_double_a = dc.reshape(self.dc_double_a, shape_a)
        dc_double_b = dc.reshape(self.dc_double_b, shape_b)
        dc_double_c = dc.reshape(self.dc_double_c, shape_c)
        npr = temp_gemm(np_double_a, np_double_b, np_double_c, self.alpha, self.beta, transA, transB)
        dcr = dc.gemm(dc_double_a, dc_double_b, dc_double_c, self.alpha, self.beta, transA, transB)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.double),
                rtol=1e-3, atol=1e-3)
        
    def test_Gemm2D_double_4 (self):
        shape_a = (6,8) 
        shape_b = (8,6) 
        shape_c = (8,8) 
        transA = 1
        transB = 1
        np_double_a = np.reshape(self.np_double_a, shape_a)
        np_double_b = np.reshape(self.np_double_b, shape_b)
        np_double_c = np.reshape(self.np_double_c, shape_c)
        dc_double_a = dc.reshape(self.dc_double_a, shape_a)
        dc_double_b = dc.reshape(self.dc_double_b, shape_b)
        dc_double_c = dc.reshape(self.dc_double_c, shape_c)
        npr = temp_gemm(np_double_a, np_double_b, np_double_c, self.alpha, self.beta, transA, transB)
        dcr = dc.gemm(dc_double_a, dc_double_b, dc_double_c, self.alpha, self.beta, transA, transB)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.double),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
