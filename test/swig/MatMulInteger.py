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

class MatMulIntegerTest(unittest.TestCase):
    def setUp(self):
        self.len = 12
        self.np_a = np.random.randint(12, size=self.len)
        self.np_b = np.random.randint(12, size=self.len)      
        self.dc_a = dc.array(list(self.np_a)).asTypeInt(); 
        self.dc_b = dc.array(list(self.np_b)).asTypeInt();
        self.a_zero_point = dc.array(0).asTypeInt();
        self.b_zero_point = dc.array(0).asTypeInt();

    def test_MatMulInteger1D (self):
        npr = np.matmul(self.np_a, self.np_b)    
        dcr = dc.matmulinteger(self.dc_a, self.dc_b,self.a_zero_point, self.b_zero_point)
        np.testing.assert_array_equal(npr, np.array(dcr.data()[0]))

    def test_MatMulInteger2D (self):
        np_a = np.reshape(self.np_a, (3,4))
        np_b = np.reshape(self.np_b, (4,3))
        dc_a = dc.reshape(self.dc_a, (3,4));
        dc_b = dc.reshape(self.dc_b, (4,3));
        npr = np.matmul(np_a, np_b);
        dcr = dc.matmulinteger(dc_a, dc_b,self.a_zero_point,self.b_zero_point);
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
                rtol=1e-3, atol=1e-3)

    def test_MatMulInteger3D (self):
        np_a = np.reshape(self.np_a, (2,2,3))
        np_b = np.reshape(self.np_b, (2,3,2))
        dc_a = dc.reshape(self.dc_a, (2,2,3));
        dc_b = dc.reshape(self.dc_b, (2,3,2));

        npr = np.matmul(np_a, np_b);
        dcr = dc.matmulinteger(dc_a, dc_b,self.a_zero_point,self.b_zero_point);

        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.int),
                rtol=1e-3, atol=1e-3)
    
    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
    
