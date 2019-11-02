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

class GreaterTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randint(24, size=self.len)
        self.np_b = np.random.randint(24, size=self.len)
        self.dc_a = dc.array(list(self.np_a))
        self.dc_b = dc.array(list(self.np_b))

    def test_Greater1D (self):
        npr = np.greater(self.np_a, self.np_b)
        dcr = dc.greater(self.dc_a, self.dc_b)
        np.testing.assert_array_equal(npr, np.array(dcr.data()))

    def test_Greater2D_1 (self):
        np_a = np.reshape(self.np_a, (6,4))
        np_b = np.reshape(self.np_b, (6,4))
        dc_a = dc.reshape(self.dc_a, (6,4))
        dc_b = dc.reshape(self.dc_b, (6,4))
        npr = np.greater(np_a, np_b)
        dcr = dc.greater(dc_a, dc_b)
        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_Greater2D_2 (self):
        np_a = np.reshape(self.np_a, (3,8))
        np_b = np.reshape(self.np_b, (3,8))
        dc_a = dc.reshape(self.dc_a, (3,8))
        dc_b = dc.reshape(self.dc_b, (3,8))
        npr = np.greater(np_a, np_b)
        dcr = dc.greater(dc_a, dc_b)
        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_Greater2D_3 (self):
        np_a = np.reshape(self.np_a, (12,2))
        np_b = np.reshape(self.np_b, (12,2))
        dc_a = dc.reshape(self.dc_a, (12,2))
        dc_b = dc.reshape(self.dc_b, (12,2))
        npr = np.greater(np_a, np_b)
        dcr = dc.greater(dc_a, dc_b)
        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_Greater3D_1 (self):
        np_a = np.reshape(self.np_a, (2,4,3))
        np_b = np.reshape(self.np_b, (2,4,3))
        dc_a = dc.reshape(self.dc_a, (2,4,3))
        dc_b = dc.reshape(self.dc_b, (2,4,3))

        npr = np.greater(np_a, np_b)
        dcr = dc.greater(dc_a, dc_b)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_Greater3D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,6))
        np_b = np.reshape(self.np_b, (2,2,6))
        dc_a = dc.reshape(self.dc_a, (2,2,6))
        dc_b = dc.reshape(self.dc_b, (2,2,6))

        npr = np.greater(np_a, np_b)
        dcr = dc.greater(dc_a, dc_b)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_Greater3D_3 (self):
        np_a = np.reshape(self.np_a, (4,1,6))
        np_b = np.reshape(self.np_b, (4,1,6))
        dc_a = dc.reshape(self.dc_a, (4,1,6))
        dc_b = dc.reshape(self.dc_b, (4,1,6))

        npr = np.greater(np_a, np_b)
        dcr = dc.greater(dc_a, dc_b)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_Greater4D_1 (self):
        np_a = np.reshape(self.np_a, (2,2,2,3))
        np_b = np.reshape(self.np_b, (2,2,2,3))
        dc_a = dc.reshape(self.dc_a, (2,2,2,3))
        dc_b = dc.reshape(self.dc_b, (2,2,2,3))

        npr = np.greater(np_a, np_b)
        dcr = dc.greater(dc_a, dc_b)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_Greater4D_2 (self):
        np_a = np.reshape(self.np_a, (1,4,2,3))
        np_b = np.reshape(self.np_b, (1,4,2,3))
        dc_a = dc.reshape(self.dc_a, (1,4,2,3))
        dc_b = dc.reshape(self.dc_b, (1,4,2,3))

        npr = np.greater(np_a, np_b)
        dcr = dc.greater(dc_a, dc_b)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
