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
import onnx

class LSTMTest(unittest.TestCase):

    def setUp(self):
        self.num_directions = 1
        self.seq_length = 1
        self.batch_size = 3
        self.input_size = 4
        self.hidden_size = 3

        self.np_x = np.random.randn(self.seq_length * self.batch_size * self.input_size).astype(np.float32)
        self.dc_x = dc.array(list(self.np_x))

        self.np_w = np.random.randn(self.num_directions * 4 * self.hidden_size * self.input_size).astype(np.float32)
        self.dc_w = dc.array(list(self.np_w))

        self.np_r = np.random.randn(self.num_directions * 4 * self.hidden_size * self.hidden_size).astype(np.float32)
        self.dc_r = dc.array(list(self.np_r))

        self.np_b = np.random.randn(self.num_directions * 8 * self.hidden_size).astype(np.float32)
        self.dc_b = dc.array(list(self.np_b))

        self.np_h = np.random.randn(self.num_directions * self.batch_size * self.hidden_size).astype(np.float32)
        self.dc_h = dc.array(list(self.np_h))

        self.np_c = np.random.randn(self.num_directions * self.batch_size * self.hidden_size).astype(np.float32)
        self.dc_c = dc.array(list(self.np_c))

        self.np_p = np.random.randn(self.num_directions * 3 * self.hidden_size).astype(np.float32)
        self.dc_p = dc.array(list(self.np_p))



        self.np_x = np.reshape(self.np_x, (self.seq_length, self.batch_size, self.input_size))
        self.dc_x = dc.reshape(self.dc_x, (self.seq_length, self.batch_size, self.input_size))

        self.np_w = np.reshape(self.np_w, (self.num_directions, 4 * self.hidden_size, self.input_size))
        self.dc_w = dc.reshape(self.dc_w, (self.num_directions, 4 * self.hidden_size, self.input_size))

        self.np_r = np.reshape(self.np_r, (self.num_directions, 4 * self.hidden_size, self.hidden_size))
        self.dc_r = dc.reshape(self.dc_r, (self.num_directions, 4 * self.hidden_size, self.hidden_size))

        self.np_b = np.reshape(self.np_b, (self.num_directions, 8 * self.hidden_size))
        self.dc_b = dc.reshape(self.dc_b, (self.num_directions, 8 * self.hidden_size))
        
        self.np_h = np.reshape(self.np_h, (self.num_directions, self.batch_size, self.hidden_size))
        self.dc_h = dc.reshape(self.dc_h, (self.num_directions, self.batch_size, self.hidden_size))

        self.np_c = np.reshape(self.np_c, (self.num_directions, self.batch_size, self.hidden_size))
        self.dc_c = dc.reshape(self.dc_c, (self.num_directions, self.batch_size, self.hidden_size))

        self.np_p = np.reshape(self.np_p, (self.num_directions, 3 * self.hidden_size))
        self.dc_p = dc.reshape(self.dc_p, (self.num_directions, 3 * self.hidden_size))

    def testAll (self):
        
        
        # print(drc)
        # np.testing.assert_allclose(self.onnx_npr_su.astype(np.float32), np.array(dcr.data()).astype(np.float32),rtol=1e-3, atol=1e-3)
        
        # print(self.dc_h)

        model = onnx.load('./parser/unit_operators/testcases/LSTM/LSTM.onnx')
        # rep = backend.prepare(model, device = 'CPU')
        xTest = np.loadtxt(fname = "swig/outputs_XWRB/weights/X_0.txt")
        xTest = dc.array(list(xTest))
        xTest = dc.reshape(xTest, (self.seq_length, self.batch_size, self.input_size))

        wTest = np.loadtxt(fname = "swig/outputs_XWRB/weights/W_0.txt")
        wTest = dc.array(list(wTest))
        wTest = dc.reshape(wTest, (self.num_directions, 4 * self.hidden_size, self.input_size))
        print("W")
        print(wTest[0])
        print()

        rTest = np.loadtxt(fname = "swig/outputs_XWRB/weights/R_0.txt")
        rTest = dc.array(list(rTest))

        rTest = dc.reshape(rTest, (self.num_directions, 4 * self.hidden_size, self.hidden_size))
        # print("R")
        # print(rTest[0])
        # print()

        bTest = np.loadtxt(fname = "swig/outputs_XWRB/weights/B_0.txt")
        bTest = dc.array(list(bTest))
        bTest = dc.reshape(bTest, (self.num_directions, 8 * self.hidden_size))
        # print("B")
        # print(bTest[0])
        # print()

        sTest = np.loadtxt(fname = "swig/outputs_XWRB/weights/S_0.txt")
        sTest = dc.array(list(sTest))
        sTest = dc.reshape(sTest, (self.batch_size))
        # print("S")
        # print(sTest)
        # print()

        hTest = np.loadtxt(fname = "swig/outputs_XWRB/weights/H_0.txt")
        hTest = dc.array(list(hTest))
        hTest = dc.reshape(hTest, (self.num_directions, self.batch_size, self.hidden_size))
        # print("H")
        # print(hTest[0])
        # print()

        cTest = np.loadtxt(fname = "swig/outputs_XWRB/weights/C_0.txt")
        cTest = dc.array(list(cTest))
        cTest = dc.reshape(cTest, (self.num_directions, self.batch_size, self.hidden_size))
        print("C")
        print(cTest)
        print()

        pTest = np.loadtxt(fname = "swig/outputs_XWRB/weights/P_0.txt")
        pTest = dc.array(list(pTest))
        pTest = dc.reshape(pTest, (self.num_directions, 3*self.hidden_size))
        print("P")
        print(pTest)
        print()

        outTest = np.loadtxt(fname = "swig/outputs_XWRB/outputs/Output_0.txt")
        outTest = dc.array(list(outTest))
        
        outTest = dc.reshape(outTest, (self.seq_length, self.num_directions, self.batch_size, self.hidden_size))

        dcr = dc.lstm(xTest, wTest, rTest, bTest, sTest, hTest, cTest, pTest)

       
    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

