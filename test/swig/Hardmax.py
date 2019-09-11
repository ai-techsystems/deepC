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

class HardmaxTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a))
        self.axis = 1
        self.axis1 = 1
        self.axis2 = 1
    def coerce(self,a):
        for i in range(self.axis):
          self.axis1 *= a.shape[i]
        self.axis2 = a.size // self.axis1;

    def test_Hardmax1D (self):
        npr = (self.np_a.max(0,keepdims=1) == self.np_a)
        dcr = dc.hardmax(self.dc_a,1)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Hardmax2D_1 (self):
        np_a = np.reshape(self.np_a, (6,4))
        dc_a = dc.reshape(self.dc_a, (6,4))
        self.coerce(np_a)
        np_a = np.reshape(np_a, (self.axis1,self.axis2))
        npr = (np_a.max(0,keepdims=1)==np_a).astype(float)
        dcr = dc.hardmax(dc_a,self.axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Hardmax2D_2 (self):
        np_a = np.reshape(self.np_a, (3,8))
        dc_a = dc.reshape(self.dc_a, (3,8))
        self.coerce(np_a)
        np_a = np.reshape(np_a, (self.axis1,self.axis2))
        npr = (np_a.max(0,keepdims=1)==np_a).astype(float)
        dcr = dc.hardmax(dc_a,self.axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Hardmax2D_3 (self):
        np_a = np.reshape(self.np_a, (12,2))
        dc_a = dc.reshape(self.dc_a, (12,2))
        self.coerce(np_a)
        np_a = np.reshape(np_a, (self.axis1,self.axis2))
        npr = (np_a.max(0,keepdims=1)==np_a).astype(float)
        dcr = dc.hardmax(dc_a,self.axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Hardmax3D_1 (self):
        np_a = np.reshape(self.np_a, (2,4,3))
        dc_a = dc.reshape(self.dc_a, (2,4,3))
        self.coerce(np_a)
        np_a = np.reshape(np_a, (self.axis1,self.axis2))
        npr = (np_a.max(0,keepdims=1)==np_a).astype(float)
        dcr = dc.hardmax(dc_a,self.axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Hardmax3D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,6))
        dc_a = dc.reshape(self.dc_a, (2,2,6))
        self.coerce(np_a)
        np_a = np.reshape(np_a, (self.axis1,self.axis2))
        npr = (np_a.max(0,keepdims=1)==np_a).astype(float)
        dcr = dc.hardmax(dc_a,self.axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Hardmax4D_1 (self):
        np_a = np.reshape(self.np_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_a, (2,2,2,3))
        self.coerce(np_a)
        np_a = np.reshape(np_a, (self.axis1,self.axis2))
        npr = (np_a.max(0,keepdims=1)==np_a).astype(float)
        dcr = dc.hardmax(dc_a,self.axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Hardmax4D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,1,6))
        dc_a = dc.reshape(self.dc_a, (2,2,1,6))
        self.coerce(np_a)
        np_a = np.reshape(np_a, (self.axis1,self.axis2))
        npr = (np_a.max(0,keepdims=1)==np_a).astype(float)
        dcr = dc.hardmax(dc_a,self.axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
