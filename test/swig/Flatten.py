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

def temp_flatten(x, shape, axis):
    new_shape = (1, -1) if axis == 0 else (np.prod(shape[0:axis]).astype(int), -1)
    y = np.reshape(x, new_shape)
    return y

class FlattenTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a))

    def test_Flatten1D (self):
        shape = (24,)
        axis = 0
        npr = temp_flatten(self.np_a,shape,axis)
        dcr = dc.flatten(self.dc_a,axis)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)
    
    def test_Flatten2D (self):
        shape = (6,4)
        axis = 1
        np_a = np.reshape(self.np_a, shape)
        dc_a = dc.reshape(self.dc_a, shape)
        npr = temp_flatten(np_a,shape,axis)
        dcr = dc.flatten(dc_a,axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Flatten3D (self):
        shape = (2,4,3)
        axis = 2
        np_a = np.reshape(self.np_a, shape)
        dc_a = dc.reshape(self.dc_a, shape)
        npr = temp_flatten(np_a,shape,axis)
        dcr = dc.flatten(dc_a,axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)
    
    def test_Flatten4D (self):
        shape = (2,2,2,3)
        axis = 4
        np_a = np.reshape(self.np_a, shape)
        dc_a = dc.reshape(self.dc_a, shape)
        npr = temp_flatten(np_a,shape,axis)
        dcr = dc.flatten(dc_a,axis)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
    
