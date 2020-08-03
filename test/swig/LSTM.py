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

class LSTMTest(unittest.TestCase):

    def setUp(self, numdirections, seqlength, inputsize, batchsize, hiddensize):
        self.num_directions = numdirections
        self.seq_length = seqlength
        self.input_size = inputsize
        self.batch_size = batchsize
        self.hidden_size = hiddensize

        self.np_x = np.random.randn(self.seq_length * self.batch_size * self.input_size).astype(np.float32)
        self.dc_x = dc.array(list(self.np_x))

        self.np_w = np.random.randn(self.num_directions * 4 * self.hidden_dize * self.input_size).astype(np.float32)
        self.dc_w = dc.array(list(self.np_w))

        self.np_r = np.random.randn(self.num_directions * 4 * self.hidden_size * self.hidden_size).astype(np.float32)
        self.dc_r = dc.array(list(self.np_r))

        self.np_h = np.random.randn(self.num_directions * self.batch_size * self.hidden_size).astype(np.float32)
        self.dc_h = dc.array(list(self.np_h))

        self.np_c = np.random.randn(self.num_directions * self.batch_size * self.hidden_size).astype(np.float32)
        self.dc_c = dc.array(list(self.np_c))

        self.np_p = np.random.randn(self.num_directions * 3 * hidden_size).astype(np.float32)
        self.dc_p = dc.array(list(self.np_p))

    def testAll (self):
        
        setUp(1, 1, 4, 3, 5)
        dcr = dc.lstm(self.onnx_dc_X, self.onnx_dc_W,  self.dc_B, "SAME_UPPER")
        # np.testing.assert_allclose(self.onnx_npr_su.astype(np.float32), np.array(dcr.data()).astype(np.float32),rtol=1e-3, atol=1e-3)


        input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

        input_size = 2
        hidden_size = 3
        weight_scale = 0.1
        number_of_gates = 4

        node = onnx.helper.make_node(
        'LSTM',
            inputs=['X', 'W', 'R'],
            outputs=['', 'Y'],
            hidden_size=hidden_size
        )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

        lstm = LSTM_Helper(X=input, W=W, R=R)
        _, Y_h = lstm.step()
        expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_lstm_defaults')
   
   
    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

