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

class SliceTest(unittest.TestCase):
    def setUp(self):
        self.len = 4096
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a));


    def test_Slice1D (self):
        len   = self.len
        np_a  = self.np_a
        dc_a  = self.dc_a
        np_start = np.random.randint(len/2)
        np_end   = np.random.randint(len/2 + 1, len)
        np_step  = np.random.randint(1, len-1)
        np_axes  = 0
        dc_start = dc.array(1).asTypeULong()
        dc_end   = dc.array(1).asTypeULong()
        dc_axes  = dc.array(1).asTypeInt()
        dc_step  = dc.array(1).asTypeULong()
        dc_start[0] = np_start
        dc_end[0]   = np_end
        dc_axes[0]  = np_axes
        dc_step[0]  = np_step
        npr   = np_a[np_start:np_end:np_step]
        dcr   = dc.slice(dc_a, dc_start, dc_end, dc_axes, dc_step)
        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)


    def test_Slice2D (self):
        len   = np.sqrt(self.len).astype(int)
        np_a  = np.reshape(self.np_a, (len, len))
        # dc_a  = dc.reshape(self.dc_a, (len, len)) $ BUG (reported bug.py) - for now using a WA
        dc_a  = dc.reshape(self.dc_a, (64, 64))
        np_start  = np.random.randint(len/2)
        np_end    = np.random.randint(len/2 + 1, len)
        np_step   = np.random.randint(1, len-1)
        np_axes   = 0
        np_start2 = np.random.randint(len/2)
        np_end2   = np.random.randint(len/2 + 1, len)
        np_step2  = np.random.randint(1, len-1)
        np_axes2  = 1
        dc_start  = dc.array(2).asTypeULong()
        dc_end    = dc.array(2).asTypeULong()
        dc_axes   = dc.array(2).asTypeInt()
        dc_step   = dc.array(2).asTypeULong()
        dc_start[0] = np_start
        dc_end[0]   = np_end
        dc_axes[0]  = np_axes
        dc_step[0]  = np_step
        dc_start[1] = np_start2
        dc_end[1]   = np_end2
        dc_axes[1]  = np_axes2
        dc_step[1]  = np_step2
        npr   = np_a[np_start:np_end:np_step, np_start2:np_end2:np_step2]
        dcr   = dc.slice(dc_a, dc_start, dc_end, dc_axes, dc_step)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)


    def test_Slice3D (self):
        len   = np.cbrt(self.len).astype(int)
        np_a  = np.reshape(self.np_a, (len, len, len))
        # dc_a  = dc.reshape(self.dc_a, (len, len, len))
        dc_a  = dc.reshape(self.dc_a, (16, 16, 16))
        np_start  = np.random.randint(len/2)
        np_end    = np.random.randint(len/2 + 1, len)
        np_step   = np.random.randint(1, len-1)
        np_axes   = 0
        np_start2 = np.random.randint(len/2)
        np_end2   = np.random.randint(len/2 + 1, len)
        np_step2  = np.random.randint(1, len-1)
        np_axes2  = 1
        np_start3 = np.random.randint(len/2)
        np_end3   = np.random.randint(len/2 + 1, len)
        np_step3  = np.random.randint(1, len-1)
        np_axes3  = 2
        dc_start  = dc.array(3).asTypeULong()
        dc_end    = dc.array(3).asTypeULong()
        dc_axes   = dc.array(3).asTypeInt()
        dc_step   = dc.array(3).asTypeULong()
        dc_start[0] = np_start
        dc_end[0]   = np_end
        dc_axes[0]  = np_axes
        dc_step[0]  = np_step
        dc_start[1] = np_start2
        dc_end[1]   = np_end2
        dc_axes[1]  = np_axes2
        dc_step[1]  = np_step2
        dc_start[2] = np_start3
        dc_end[2]   = np_end3
        dc_axes[2]  = np_axes3
        dc_step[2]  = np_step3
        npr   = np_a[np_start:np_end:np_step, np_start2:np_end2:np_step2, np_start3:np_end3:np_step3]
        dcr   = dc.slice(dc_a, dc_start, dc_end, dc_axes, dc_step)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def test_Slice4D (self):
        len   = np.sqrt(np.sqrt(self.len)).astype(int)
        np_a  = np.reshape(self.np_a, (len, len, len, len))
        # dc_a  = dc.reshape(self.dc_a, (len, len, len))
        dc_a  = dc.reshape(self.dc_a, (8, 8, 8, 8))
        np_start  = np.random.randint(len/2)
        np_end    = np.random.randint(len/2 + 1, len)
        np_step   = np.random.randint(1, len-1)
        np_axes   = 0
        np_start2 = np.random.randint(len/2)
        np_end2   = np.random.randint(len/2 + 1, len)
        np_step2  = np.random.randint(1, len-1)
        np_axes2  = 1
        np_start3 = np.random.randint(len/2)
        np_end3   = np.random.randint(len/2 + 1, len)
        np_step3  = np.random.randint(1, len-1)
        np_axes3  = 2
        np_start4 = np.random.randint(len/2)
        np_end4   = np.random.randint(len/2 + 1, len)
        np_step4  = np.random.randint(1, len-1)
        np_axes4  = 3
        dc_start  = dc.array(4).asTypeULong()
        dc_end    = dc.array(4).asTypeULong()
        dc_axes   = dc.array(4).asTypeInt()
        dc_step   = dc.array(4).asTypeULong()
        dc_start[0] = np_start
        dc_end[0]   = np_end
        dc_axes[0]  = np_axes
        dc_step[0]  = np_step
        dc_start[1] = np_start2
        dc_end[1]   = np_end2
        dc_axes[1]  = np_axes2
        dc_step[1]  = np_step2
        dc_start[2] = np_start3
        dc_end[2]   = np_end3
        dc_axes[2]  = np_axes3
        dc_step[2]  = np_step3
        dc_start[3] = np_start4
        dc_end[3]   = np_end4
        dc_axes[3]  = np_axes4
        dc_step[3]  = np_step4
        npr   = np_a[np_start:np_end:np_step, np_start2:np_end2:np_step2, np_start3:np_end3:np_step3, np_start4:np_end4:np_step4]
        dcr   = dc.slice(dc_a, dc_start, dc_end, dc_axes, dc_step)
        np.testing.assert_allclose(npr.flatten(), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

