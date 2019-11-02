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

import os,sys
# DNNC_ROOT='/Desktop/dnnCompiler'
sys.path.append(os.path.abspath('..'+os.path.sep+'..'+os.path.sep+'deepC'));

import deepC.dnnc as dc
import numpy as np

import unittest



class LpNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a));
        self.axis = 1
        self.p = 2

    def norm_2(self,x):
        return x/np.linalg.norm(x,ord=self.p)
    def norm_1(self,x):
        return x/np.linalg.norm(x,ord=1)

    def test_LpNormalization2D_1 (self):
        np_a = np.reshape(self.np_a, (4,6))
        dc_a = dc.reshape(self.dc_a, (4,6));
        npr = np.apply_along_axis(self.norm_2,self.axis,np_a)
        dcr = dc.lpnormalization(dc_a, self.p, self.axis);
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_LpNormalization2D_2 (self):
        np_a = np.reshape(self.np_a, (12,2))
        dc_a = dc.reshape(self.dc_a, (12,2));
        axis = 0
        p = 2
        npr = np.apply_along_axis(self.norm_2,axis,np_a)
        dcr = dc.lpnormalization(dc_a, p, axis);
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_LpNormalization2D_3 (self):
        np_a = np.reshape(self.np_a, (8,3))
        dc_a = dc.reshape(self.dc_a, (8,3));
        axis = 0
        p = 1
        npr = np.apply_along_axis(self.norm_1,axis,np_a)
        dcr = dc.lpnormalization(dc_a, p, axis);
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_LpNormalization2D_4 (self):
        np_a = np.reshape(self.np_a, (8,3))
        dc_a = dc.reshape(self.dc_a, (8,3));
        axis = 1
        p = 1
        npr = np.apply_along_axis(self.norm_1,axis,np_a)
        dcr = dc.lpnormalization(dc_a, p, axis);
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

