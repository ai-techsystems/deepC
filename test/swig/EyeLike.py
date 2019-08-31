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


class EyeLikeTest(unittest.TestCase):
    
    def setUp(self):
        self.len = 10
        # self.k = np.random.randint(low=-10, high=10)
        self.k = 1
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a))


    # EyeLike by default takes 2D tensor only

    def test_EyeLike2D (self):
        row = 2
        column = 5
        np_a = np.reshape(self.np_a, (row,column))
        dc_a = dc.reshape(self.dc_a, (row,column))
        npr = np.eye(row, column ,k=self.k)
        dcr = dc.eye_like(dc_a,self.k)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    # unittest.main()
    
    k = 2
    
    len = 10
    
    np_a = np.random.randn(len).astype(np.int8)
    dc_a = dc.array(list(np_a))
    
    # print(np_a)
    # print(dc_a)
    
    row = 2
    column = 5
    
    np_b = np.reshape(np_a, (row,column))
    dc_b = dc.reshape(dc_a, (row,column))
    
    print(np_b)
    print(np.array(dc_b.data()))
    
    # print(np_a)
    # print(dc_a)
    
    npr = np.eye(row, column ,k=k)
    dcr = dc.eye_like(dc_b,k)
    print(npr.flatten())
    print(np.array(dcr.data()))
    print(npr)
    print(dcr)
    
