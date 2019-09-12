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

import common; # DNNC path setup

import dnnc as dc
import numpy as np
import unittest



class MinTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.np_b = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a));
        
        self.dc_b = dc.array(list(self.np_b));
        
    def test_Min1D (self):
        npr = np.minimum(self.np_a, self.np_b)
        dcr = dc.min(dc.ftvec([self.dc_a,self.dc_b]))
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Min2D (self):
        np_a = np.reshape(self.np_a, (6,4))
        np_b = np.reshape(self.np_b, (6,4))
        dc_a = dc.reshape(self.dc_a, (6,4));
        dc_b = dc.reshape(self.dc_b, (6,4));
        npr = np.minimum(np_a, np_b)
        dcr = dc.min(dc.ftvec([dc_a,dc_b]));
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Min3D (self):
        np_a = np.reshape(self.np_a, (2,4,3))
        np_b = np.reshape(self.np_b, (2,4,3))
        dc_a = dc.reshape(self.dc_a, (2,4,3));
        dc_b = dc.reshape(self.dc_b, (2,4,3));

        npr = np.minimum(np_a, np_b)
        dcr = dc.min(dc.ftvec([dc_a,dc_b]));
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Min4D (self):
        np_a = np.reshape(self.np_a, (2,2,3,2))
        np_b = np.reshape(self.np_b, (2,2,3,2))
        dc_a = dc.reshape(self.dc_a, (2,2,3,2));
        dc_b = dc.reshape(self.dc_b, (2,2,3,2));

        npr = np.minimum(np_a, np_b)
        dcr = dc.min(dc.ftvec([dc_a,dc_b]));
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
    
