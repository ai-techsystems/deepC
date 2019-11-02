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

import common

import deepC.dnnc as dc
import numpy as np
import unittest
import sys

class BroadcastTest(unittest.TestCase):
    def setUp(self):
        self.len1 = 20
        self.len2 = 40
        self.np_a = np.random.randn(self.len1).astype(np.float32)
        self.np_b = np.random.randn(self.len2).astype(np.float32)
        self.np_c = np.random.randn(self.len1).astype(np.float32)
        self.np_d = np.random.randn(self.len2).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a));
        self.dc_b = dc.array(list(self.np_b));
        self.dc_c = dc.array(list(self.np_c));
        self.dc_d = dc.array(list(self.np_d));
        self.err  = "operands could not be broadcast together with shapes (2,5,2) (4,5,2)   ";

    def test_error_message (self):
        dc_a = dc.reshape(self.dc_a, (2,5,2));
        dc_b = dc.reshape(self.dc_b, (4,5,2));
        np_a = np.reshape(self.np_a,(2,5,2))
        np_b = np.reshape(self.np_b,(4,5,2))
        try:
            np_sum = np.add(np_a,np_b)
        except:
            type, val, tb = sys.exc_info()
            np_err = val.__str__()
            assert (np_err[0:65]==self.err[0:65]), "ASSERT FAILED for numpy error message"
        try:
            dc_sum = dc.add(dc_a,dc_b)
        except:
            type, val, tb = sys.exc_info()
            dc_err = val.__str__()
            assert (dc_err[0:65]==self.err[0:65]), "ASSERT FAILED for dc error message"
            assert (dc_err[0:65]==np_err[0:65]),   "ASSERT FAILED for matching numpy and dc error message"

    def test_Add(self):
        dc_a = dc.reshape(self.dc_a, (5,4));
        dc_b = dc.reshape(self.dc_b, (2,5,4));
        np_a = np.reshape(self.np_a,(5,4))
        np_b = np.reshape(self.np_b,(2,5,4))
        npr = np.add(np_a, np_b);
        dcr = dc.add(dc_a, dc_b);
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)


    def test_Maxof4(self):
       np_a = np.reshape(self.np_a,(5,4))
       np_b = np.reshape(self.np_b,(2,5,4))
       np_c = np.reshape(self.np_c,(5,4))
       np_d = np.reshape(self.np_d,(2,5,4))
       npr  = np.maximum(np.maximum(np_a,np_b),np.maximum(np_c,np_d))
       dc_a = dc.reshape(self.dc_a,(5,4))
       dc_b = dc.reshape(self.dc_b,(2,5,4))
       dc_c = dc.reshape(self.dc_c,(5,4))
       dc_d = dc.reshape(self.dc_d,(2,5,4))
       dcr  = dc.max(dc.vectorTensorFloat([dc_a, dc_b, dc_c, dc_d]))
       np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)


    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

