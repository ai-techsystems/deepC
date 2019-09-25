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

def temp_dropout(x, ratio):
    y = x
    return y

class DropoutTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.ratio = 2.0
        
        self.np_float_a = np.random.randn(self.len).astype(np.float32)
        self.dc_float_a = dc.array(list(self.np_float_a))

        self.np_double_a = np.random.randn(self.len).astype(np.float64)
        self.dc_double_a = dc.array(list(self.np_double_a))

    def test_Dropout1D_float (self):
        npr = temp_dropout(self.np_float_a,self.ratio)
        dcr = dc.dropout(self.dc_float_a,self.ratio)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)
    
    def test_Dropout1D_double (self):
        npr = temp_dropout(self.np_double_a,self.ratio)
        dcr = dc.dropout(self.dc_double_a,self.ratio)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float64),
                rtol=1e-3, atol=1e-3)
    
    def test_Dropout2D_float (self):
        np_a = np.reshape(self.np_float_a, (6,4))
        dc_a = dc.reshape(self.dc_float_a, (6,4))
        npr = temp_dropout(np_a,self.ratio)
        dcr = dc.dropout(dc_a,self.ratio)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Dropout2D_double (self):
        np_a = np.reshape(self.np_double_a, (6,4))
        dc_a = dc.reshape(self.dc_double_a, (6,4))
        npr = temp_dropout(np_a,self.ratio)
        dcr = dc.dropout(dc_a,self.ratio)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Dropout3D_float (self):
        np_a = np.reshape(self.np_float_a, (2,4,3))
        dc_a = dc.reshape(self.dc_float_a, (2,4,3))
        npr = temp_dropout(np_a,self.ratio)
        dcr = dc.dropout(dc_a,self.ratio)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Dropout3D_double (self):
        np_a = np.reshape(self.np_double_a, (2,4,3))
        dc_a = dc.reshape(self.dc_double_a, (2,4,3))
        npr = temp_dropout(np_a,self.ratio)
        dcr = dc.dropout(dc_a,self.ratio)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Dropout4D_float (self):
        np_a = np.reshape(self.np_float_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_float_a, (2,2,2,3))
        npr = temp_dropout(np_a,self.ratio)
        dcr = dc.dropout(dc_a,self.ratio)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Dropout4D_double (self):
        np_a = np.reshape(self.np_double_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_double_a, (2,2,2,3))
        npr = temp_dropout(np_a,self.ratio)
        dcr = dc.dropout(dc_a,self.ratio)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()