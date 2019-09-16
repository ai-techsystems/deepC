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

import dnnc as dc
import numpy as np
import unittest

class ModTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a_float = np.random.randn(self.len).astype(np.int32)
        self.np_b_float = np.random.randn(self.len).astype(np.int32)
        self.dc_a_float = dc.array(list(self.np_a_float));
        self.dc_b_float = dc.array(list(self.np_b_float));
        self.fmod_flag = 1

    def test_Mod1D (self):
        npr = np.mod(self.np_a_float, self.np_b_float)
        dcr = dc.neg(self.dc_a_float, self.dc_b_float, self.fmod_flag)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.int32),
                rtol=1e-3, atol=1e-3)

    def test_Mod2D (self):
        np_a_float = np.reshape(self.np_a_float, (6,4))
        np_b_float = np.reshape(self.np_b_float, (6,4))
        dc_a_float = dc.reshape(self.dc_a_float, (6,4));
        dc_b_float = dc.reshape(self.dc_b_float, (6,4));
        npr = np.mod(np_a_float, np_b_float);
        dcr = dc.mod(dc_a_float, dc_b_float, self.fmod_flag);
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int32),
                rtol=1e-3, atol=1e-3)

    def test_Mod3D (self):
        np_a_float = np.reshape(self.np_a_float, (2,4,3))
        np_b_float = np.reshape(self.np_b_float, (2,4,3))
        dc_a_float = dc.reshape(self.dc_a_float, (2,4,3));
        dc_b_float = dc.reshape(self.dc_b_float, (2,4,3));

        npr = np.mod(np_a_float, np_b_float);
        dcr = dc.mod(dc_a_float, dc_b_float, self.fmod_flag);

        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int32),
                rtol=1e-3, atol=1e-3)

    def test_Mod4D (self):
        np_a_float = np.reshape(self.np_a_float, (2,2,2,3))
        np_b_float = np.reshape(self.np_b_float, (2,2,2,3))
        dc_a_float = dc.reshape(self.dc_a_float, (2,2,2,3));
        dc_b_float = dc.reshape(self.dc_b_float, (2,2,2,3));


        npr = np.mod(np_a_float, np_b_float);
        dcr = dc.mod(dc_a_float, dc_b_float, self.fmod_flag);

        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

