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

class GlobalAveragePoolTest(unittest.TestCase):
    def setUp(self):
        self.len = 24
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a))

        def test_GlobalAveragePool2D (self):
            np_a = np.reshape(self.np_a, (4,6))
            dc_a = dc.reshape(self.dc_a, (4,6))
            if(np.ndim(np_a)<=2):
                npr = np_a
            else:
                spatial_shape = np.ndim(np_a) -2
                npr = np.average(np_a, axis=tuple(range(2, spatial_shape + 2)))
                for _ in range(spatial_shape):
                    npr = np.expand_dims(npr, -1)
            dcr = dc.global_average_pool(dc_a)
            np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                    rtol=1e-3, atol=1e-3)

    def test_GlobalAveragePool3D_1 (self):
        np_a = np.reshape(self.np_a, (2,4,3))
        dc_a = dc.reshape(self.dc_a, (2,4,3))
        spatial_shape = np.ndim(np_a) -2
        npr = np.average(np_a, axis=tuple(range(2, spatial_shape + 2)))
        for _ in range(spatial_shape):
            npr = np.expand_dims(npr, -1)
        dcr = dc.global_average_pool(dc_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_GlobalAveragePool3D_2 (self):
        np_a = np.reshape(self.np_a, (2,2,6))
        dc_a = dc.reshape(self.dc_a, (2,2,6))
        spatial_shape = np.ndim(np_a) -2
        npr = np.average(np_a, axis=tuple(range(2, spatial_shape + 2)))
        for _ in range(spatial_shape):
            npr = np.expand_dims(npr, -1)
        dcr = dc.global_average_pool(dc_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_GlobalAveragePool3D_3 (self):
        np_a = np.reshape(self.np_a, (8,1,3))
        dc_a = dc.reshape(self.dc_a, (8,1,3))
        spatial_shape = np.ndim(np_a) -2
        npr = np.average(np_a, axis=tuple(range(2, spatial_shape + 2)))
        for _ in range(spatial_shape):
            npr = np.expand_dims(npr, -1)
        dcr = dc.global_average_pool(dc_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_GlobalAveragePool4D_1 (self):
        np_a = np.reshape(self.np_a, (2,2,2,3))
        dc_a = dc.reshape(self.dc_a, (2,2,2,3))
        spatial_shape = np.ndim(np_a) -2
        npr = np.average(np_a, axis=tuple(range(2, spatial_shape + 2)))
        for _ in range(spatial_shape):
            npr = np.expand_dims(npr, -1)
        dcr = dc.global_average_pool(dc_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_GlobalAveragePool4D_2 (self):
        np_a = np.reshape(self.np_a, (2,1,2,6))
        dc_a = dc.reshape(self.dc_a, (2,1,2,6))
        spatial_shape = np.ndim(np_a) -2
        npr = np.average(np_a, axis=tuple(range(2, spatial_shape + 2)))
        for _ in range(spatial_shape):
            npr = np.expand_dims(npr, -1)
        dcr = dc.global_average_pool(dc_a)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()
