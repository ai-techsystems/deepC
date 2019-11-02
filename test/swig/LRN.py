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
import math

class LRNTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a))
        self.alpha = 0.0001
        self.beta = 0.75
        self.bias = 1.0
        self.size = 2

    def test_LRN3D_1 (self):
        np_a = np.reshape(self.np_a, (2,4,3))
        dc_a = dc.reshape(self.dc_a, (2,4,3))
        square_sum = np.zeros((2,4,3)).astype(np.float32)
        for n, c, h in np.ndindex(np_a.shape):
            square_sum[n, c, h] = sum(np_a[n,max(0, c - int(math.floor((self.size - 1) / 2))):min(5, c + int(math.ceil((self.size - 1) / 2)) + 1),h] ** 2)
        npr = np_a / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        dcr = dc.lrn(dc_a,self.size,self.alpha,self.beta,self.bias)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_LRN3D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,6))
        dc_a = dc.reshape(self.dc_a, (2,2,6))
        square_sum = np.zeros((2,2,6)).astype(np.float32)
        for n, c, h in np.ndindex(np_a.shape):
            square_sum[n, c, h] = sum(np_a[n,max(0, c - int(math.floor((self.size - 1) / 2))):min(5, c + int(math.ceil((self.size - 1) / 2)) + 1),h] ** 2)
        npr = np_a / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        dcr = dc.lrn(dc_a,self.size,self.alpha,self.beta,self.bias)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_LRN3D_3 (self):
        np_a = np.reshape(self.np_a, (2,4,3))
        dc_a = dc.reshape(self.dc_a, (2,4,3))
        square_sum = np.zeros((2,4,3)).astype(np.float32)
        self.size = 3
        for n, c, h in np.ndindex(np_a.shape):
            square_sum[n, c, h] = sum(np_a[n,max(0, c - int(math.floor((self.size - 1) / 2))):min(5, c + int(math.ceil((self.size - 1) / 2)) + 1),h] ** 2)
        npr = np_a / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        dcr = dc.lrn(dc_a,self.size,self.alpha,self.beta,self.bias)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_LRN3D_4 (self):
        np_a = np.reshape(self.np_a, (2,2,6))
        dc_a = dc.reshape(self.dc_a, (2,2,6))
        self.alpha = 0.0002
        self.beta = 0.5
        self.bias = 2.0
        self.size = 3
        square_sum = np.zeros((2,2,6)).astype(np.float32)
        for n, c, h in np.ndindex(np_a.shape):
            square_sum[n, c, h] = sum(np_a[n,max(0, c - int(math.floor((self.size - 1) / 2))):min(5, c + int(math.ceil((self.size - 1) / 2)) + 1),h] ** 2)
        npr = np_a / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        dcr = dc.lrn(dc_a,self.size,self.alpha,self.beta,self.bias)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_LRN4D_1 (self):
        np_a = np.reshape(self.np_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_a, (2,2,2,3))
        square_sum = np.zeros((2,2,2,3)).astype(np.float32)
        for n, c, h,w in np.ndindex(np_a.shape):
            square_sum[n, c, h,w] = sum(np_a[n,max(0, c - int(math.floor((self.size - 1) / 2))):min(5, c + int(math.ceil((self.size - 1) / 2)) + 1),h,w] ** 2)
        npr = np_a / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        dcr = dc.lrn(dc_a,self.size)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_LRN4D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,1,6))
        dc_a = dc.reshape(self.dc_a, (2,2,1,6))
        square_sum = np.zeros((2,2,1,6)).astype(np.float32)
        for n, c, h,w in np.ndindex(np_a.shape):
            square_sum[n, c, h,w] = sum(np_a[n,max(0, c - int(math.floor((self.size - 1) / 2))):min(5, c + int(math.ceil((self.size - 1) / 2)) + 1),h,w] ** 2)
        npr = np_a / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        dcr = dc.lrn(dc_a,self.size)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_LRN4D_3 (self):
        np_a = np.reshape(self.np_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_a, (2,2,2,3))
        square_sum = np.zeros((2,2,2,3)).astype(np.float32)
        self.alpha = 0.0002
        self.beta = 0.5
        self.bias = 2.0
        self.size = 3
        for n, c, h,w in np.ndindex(np_a.shape):
            square_sum[n, c, h,w] = sum(np_a[n,max(0, c - int(math.floor((self.size - 1) / 2))):min(5, c + int(math.ceil((self.size - 1) / 2)) + 1),h,w] ** 2)
        npr = np_a / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        dcr = dc.lrn(dc_a,self.size,self.alpha,self.beta,self.bias)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_LRN4D_4 (self):
        np_a = np.reshape(self.np_a, (2,2,1,6))
        dc_a = dc.reshape(self.dc_a, (2,2,1,6))
        self.alpha = 0.0002
        self.beta = 0.5
        self.bias = 2.0
        self.size = 3
        square_sum = np.zeros((2,2,1,6)).astype(np.float32)
        for n, c, h,w in np.ndindex(np_a.shape):
            square_sum[n, c, h,w] = sum(np_a[n,max(0, c - int(math.floor((self.size - 1) / 2))):min(5, c + int(math.ceil((self.size - 1) / 2)) + 1),h,w] ** 2)
        npr = np_a / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        dcr = dc.lrn(dc_a,self.size,self.alpha,self.beta,self.bias)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
