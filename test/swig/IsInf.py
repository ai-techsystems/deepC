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

import dnnc as dc
import numpy as np
import unittest

def Isinf(np_a,detect_positive,detect_negative):
    if (detect_positive and not(detect_negative) ):
        np_a[np_a<0] = 0
        return np.isinf(np_a)
    elif (detect_negative and not(detect_positive) ):
        np_a[np_a>0] = 0
        return np.isinf(np_a)
    elif ( not(detect_positive) and not(detect_negative) ):
        return np.zeros_like(np_a)
    else:
        return np.isinf(np_a)

class IsInfTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randn(self.len)
        self.np_a.ravel()[np.random.choice(self.np_a.size, 5, replace=False)] = np.inf
        self.np_a.ravel()[np.random.choice(self.np_a.size, 5, replace=False)] = -np.inf
        self.dc_a = dc.array(list(self.np_a))
        self.detect_positive = 0
        self.detect_negative = 1

    def test_IsInf1D (self):
        npr = Isinf(self.np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(self.dc_a,self.detect_positive,self.detect_negative)
        np.testing.assert_array_equal(npr, np.array(dcr.data()))

    def test_IsInf2D_1 (self):
        np_a = np.reshape(self.np_a, (6,4))
        dc_a = dc.reshape(self.dc_a, (6,4))
        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)
        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsInf2D_2 (self):
        np_a = np.reshape(self.np_a, (3,8))
        dc_a = dc.reshape(self.dc_a, (3,8))
        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)
        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsInf2D_3 (self):
        np_a = np.reshape(self.np_a, (12,2))
        dc_a = dc.reshape(self.dc_a, (12,2))
        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)
        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsInf3D_1 (self):
        np_a = np.reshape(self.np_a, (2,4,3))
        dc_a = dc.reshape(self.dc_a, (2,4,3))

        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsInf3D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,6))
        dc_a = dc.reshape(self.dc_a, (2,2,6))

        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsInf3D_3 (self):
        np_a = np.reshape(self.np_a, (4,2,3))
        dc_a = dc.reshape(self.dc_a, (4,2,3))

        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsInf3D_4 (self):
        np_a = np.reshape(self.np_a, (4,2,3))
        dc_a = dc.reshape(self.dc_a, (4,2,3))
        self.detect_positive = 1
        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsInf3D_5 (self):
        np_a = np.reshape(self.np_a, (4,2,3))
        dc_a = dc.reshape(self.dc_a, (4,2,3))
        self.detect_positive = 1
        self.detect_negative = 0
        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsInf4D_1 (self):
        np_a = np.reshape(self.np_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_a, (2,2,2,3))

        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsInf4D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,1,6))
        dc_a = dc.reshape(self.dc_a, (2,2,1,6))

        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def test_IsInf4D_3 (self):
        np_a = np.reshape(self.np_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_a, (2,2,2,3))
        self.detect_positive = 1
        self.detect_negative = 0
        npr = Isinf(np_a,self.detect_positive,self.detect_negative)
        dcr = dc.isinf(dc_a,self.detect_positive,self.detect_negative)

        np.testing.assert_array_equal(npr.flatten(), np.array(dcr.data()))

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
