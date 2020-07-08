# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
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

import common # DNNC path setup

import deepC.dnnc as dc
import numpy as np
import unittest



class ReduceL1Test(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a))

    def test_ReduceL1_1D (self):

        npr = np.sum(np.abs(self.np_a), axis=(0))
        dcr = dc.reduce_L1(self.dc_a, [0])

        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

        npr2 = np.sum(np.abs(self.np_a), axis=(0))
        dcr2 = dc.reduce_L1(self.dc_a, [])
        np.testing.assert_allclose(npr2.flatten(), np.array(dcr2.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

        npr3 = np.sum(np.abs(self.np_a), axis=(0), keepdims = True)
        dcr3 = dc.reduce_L1(self.dc_a, [0], 1)
        np.testing.assert_allclose(npr3.flatten(), np.array(dcr3.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)


    def test_ReduceL1_2D (self):
        np_a = np.reshape(self.np_a, (6,4))
        dc_a = dc.reshape(self.dc_a, (6,4))

        npr = np.sum(np.abs(np_a), axis=(0))
        dcr = dc.reduce_L1(dc_a, [0])

        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

        npr2 = np.sum(np.abs(np_a), axis=(0, 1))
        dcr2 = dc.reduce_L1(dc_a, [])
        np.testing.assert_allclose(npr2.flatten(), np.array(dcr2.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

        npr3 = np.sum(np.abs(np_a), axis=(0), keepdims = True)
        dcr3 = dc.reduce_L1(dc_a, [0], 1)
        np.testing.assert_allclose(npr3.flatten(), np.array(dcr3.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)


    def test_ReduceL2_3D (self):
        np_a = np.reshape(self.np_a, (2,4,3))
        dc_a = dc.reshape(self.dc_a, (2,4,3))

        npr = np.sum(np.abs(np_a), axis=(0))
        dcr = dc.reduce_L1(dc_a, [0])

        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

        npr2 = np.sum(np.abs(np_a), axis=(0, 2))
        dcr2 = dc.reduce_L1(dc_a, [0, 2])
        np.testing.assert_allclose(npr2.flatten(), np.array(dcr2.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

        npr3 = np.sum(np.abs(np_a), axis=(0, 1, 2))
        dcr3 = dc.reduce_L1(dc_a, [])
        np.testing.assert_allclose(npr3.flatten(), np.array(dcr3.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

        npr4 = np.sum(np.abs(np_a), axis=(0), keepdims = True)
        dcr4 = dc.reduce_L1(dc_a, [0], 1)
        np.testing.assert_allclose(npr4.flatten(), np.array(dcr4.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)


    def test_ReduceL1_4D (self):
        np_a = np.reshape(self.np_a, (2,2,3,2))
        dc_a = dc.reshape(self.dc_a, (2,2,3,2))

        npr = np.sum(np.abs(np_a), axis=(0))
        dcr = dc.reduce_L1(dc_a, [0])
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

        npr2 = np.sum(np.abs(np_a), axis=(0, 2))
        dcr2 = dc.reduce_L1(dc_a, [0, 2])
        np.testing.assert_allclose(npr2.flatten(), np.array(dcr2.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

        npr3 = np.sum(np.abs(np_a), axis=(0, 1, 2, 3))
        dcr3 = dc.reduce_L1(dc_a, [])
        np.testing.assert_allclose(npr3.flatten(), np.array(dcr3.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

        npr4 = np.sum(np.abs(np_a), axis=(0), keepdims = True)
        dcr4 = dc.reduce_L1(dc_a, [0], 1)
        np.testing.assert_allclose(npr4.flatten(), np.array(dcr4.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)


    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

