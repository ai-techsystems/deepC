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

class InstanceNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a))
        self.scale = np.random.randn(4).astype(np.float32)
        self.B = np.random.randn(4).astype(np.float32)
        self.epsilon=1e-05

    def test_InstanceNormalization3D_1 (self):
        np_a = np.reshape(self.np_a, (2,4,3))
        dc_a = dc.reshape(self.dc_a, (2,4,3))
        np_a_mean = np_a.mean(axis=(0, 2), keepdims=True)
        np_a_var = np_a.var(axis=(0, 2), keepdims=True)
        scale = np.reshape(self.scale,(1,4,1))
        B = np.reshape(self.B,(1,4,1))
        npr = scale*(np_a-np_a_mean)/np.sqrt(np_a_var+self.epsilon) + B
        dcr = dc.instancenormalization(dc_a,dc.reshape(dc.array(list(self.scale)),(4,1)),dc.reshape(dc.array(list(self.B)),(4,1)),self.epsilon)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_InstanceNormalization3D_2 (self):
        np_a = np.reshape(self.np_a, (4,2,3))
        dc_a = dc.reshape(self.dc_a, (4,2,3))
        np_a_mean = np_a.mean(axis=(0, 2), keepdims=True)
        np_a_var = np_a.var(axis=(0, 2), keepdims=True)
        scale = np.reshape(self.scale[0:2],(1,2,1))
        B = np.reshape(self.B[0:2],(1,2,1))
        npr = scale*(np_a-np_a_mean)/np.sqrt(np_a_var+self.epsilon) + B
        dcr = dc.instancenormalization(dc_a,dc.reshape(dc.array(list(self.scale[0:2])),(2,1)),dc.reshape(dc.array(list(self.B[0:2])),(2,1)),self.epsilon)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_InstanceNormalization3D_3 (self):
        np_a = np.reshape(self.np_a, (2,2,6))
        dc_a = dc.reshape(self.dc_a, (2,2,6))
        np_a_mean = np_a.mean(axis=(0, 2), keepdims=True)
        np_a_var = np_a.var(axis=(0, 2), keepdims=True)
        scale = np.reshape(self.scale[0:2],(1,2,1))
        B = np.reshape(self.B[0:2],(1,2,1))
        npr = scale*(np_a-np_a_mean)/np.sqrt(np_a_var+self.epsilon) + B
        dcr = dc.instancenormalization(dc_a,dc.reshape(dc.array(list(self.scale[0:2])),(2,1)),dc.reshape(dc.array(list(self.B[0:2])),(2,1)),self.epsilon)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_InstanceNormalization4D_1 (self):
        np_a = np.reshape(self.np_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_a, (2,2,2,3))
        scale = np.reshape(self.scale[0:2],(1,2,1,1))
        B = np.reshape(self.B[0:2],(1,2,1,1))
        np_a_mean = np_a.mean(axis=(0,2,3), keepdims=True)
        np_a_var = np_a.var(axis=(0,2,3), keepdims=True)
        npr = scale*(np_a-np_a_mean)/np.sqrt(np_a_var+self.epsilon) + B
        dcr = dc.instancenormalization(dc_a,dc.reshape(dc.array(list(self.scale[0:2])),(2)),dc.reshape(dc.array(list(self.B[0:2])),(2)),self.epsilon)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_InstanceNormalization4D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_a, (2,2,2,3))
        scale = np.reshape(self.scale[0:2],(1,2,1,1))
        B = np.reshape(self.B[0:2],(1,2,1,1))
        np_a_mean = np_a.mean(axis=(0,2,3), keepdims=True)
        np_a_var = np_a.var(axis=(0,2,3), keepdims=True)
        npr = scale*(np_a-np_a_mean)/np.sqrt(np_a_var+self.epsilon) + B
        dcr = dc.instancenormalization(dc_a,dc.reshape(dc.array(list(self.scale[0:2])),(2)),dc.reshape(dc.array(list(self.B[0:2])),(2)),self.epsilon)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
