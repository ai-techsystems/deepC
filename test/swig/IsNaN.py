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

class IsNaNTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randn(self.len)
        self.np_a.ravel()[np.random.choice(self.np_a.size, 5, replace=False)] = np.nan
        self.dc_a = dc.array(list(self.np_a))

    def test_IsNaN1D (self):
        npr = np.isnan(self.np_a)
        dcr = dc.isnan(self.dc_a)
        np.testing.assert_array_equal(npr, np.array(dcr.data()))

    def test_IsNaN2D_1 (self):
        np_a = np.reshape(self.np_a, (6,4))
        dc_a = dc.reshape(self.dc_a, (6,4))
        npr = np.isnan(np_a)
        dcr = dc.isnan(dc_a)
        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsNaN2D_2 (self):
        np_a = np.reshape(self.np_a, (3,8))
        dc_a = dc.reshape(self.dc_a, (3,8))
        npr = np.isnan(np_a)
        dcr = dc.isnan(dc_a)
        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsNaN2D_3 (self):
        np_a = np.reshape(self.np_a, (12,2))
        dc_a = dc.reshape(self.dc_a, (12,2))
        npr = np.isnan(np_a)
        dcr = dc.isnan(dc_a)
        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsNaN3D_1 (self):
        np_a = np.reshape(self.np_a, (2,4,3))
        dc_a = dc.reshape(self.dc_a, (2,4,3))

        npr = np.isnan(np_a)
        dcr = dc.isnan(dc_a)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsNaN3D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,6))
        dc_a = dc.reshape(self.dc_a, (2,2,6))

        npr = np.isnan(np_a)
        dcr = dc.isnan(dc_a)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsNaN3D_3 (self):
        np_a = np.reshape(self.np_a, (4,2,3))
        dc_a = dc.reshape(self.dc_a, (4,2,3))

        npr = np.isnan(np_a)
        dcr = dc.isnan(dc_a)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsNaN4D_1 (self):
        np_a = np.reshape(self.np_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_a, (2,2,2,3))

        npr = np.isnan(np_a)
        dcr = dc.isnan(dc_a)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsNaN4D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,1,6))
        dc_a = dc.reshape(self.dc_a, (2,2,1,6))

        npr = np.isnan(np_a)
        dcr = dc.isnan(dc_a)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
