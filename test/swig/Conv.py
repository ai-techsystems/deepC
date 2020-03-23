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

class ConvTest(unittest.TestCase):
    def setUp(self):
        ## random testcase
        self.channels      = 1
        self.featuremaps   = 1
        self.batchsize     = 1
        #self.oneK          = 1024
        self.oneK          = 50
        self.X_h           = self.oneK + np.random.randint(self.oneK*3)
        self.X_w           = self.oneK + np.random.randint(self.oneK*3)
        self.K_h           = 3 + np.random.randint(97)
        self.K_w           = 3 + np.random.randint(97)
        self.np_strides    = np.zeros(2).astype(np.float32)
        self.np_strides[0] = 1 + np.random.randint(self.K_w - 1)
        self.np_strides[1] = 1 + np.random.randint(self.K_h - 1)
        self.np_B          = np.zeros(self.featuremaps).astype(np.float32)
        self.np_X_data     = np.random.randn(self.X_w * self.X_h).astype(np.float32)
        self.np_K_data     = np.random.randn(self.K_w * self.K_h).astype(np.float32)
        self.np_X          = np.reshape(self.np_X_data, (self.X_h, self.X_w))
        self.np_K          = np.reshape(self.np_K_data, (self.K_h, self.K_w))

        self.dc_X          = dc.reshape(dc.array(list(self.np_X_data)), (self.batchsize, self.channels,   self.X_h, self.X_w)).asTypeFloat()
        self.dc_K          = dc.reshape(dc.array(list(self.np_K_data)), (self.featuremaps, self.channels, self.K_h, self.K_w)).asTypeFloat()
        self.dc_B          = dc.zeros(self.featuremaps).asTypeFloat()
        self.dc_strides    = dc.reshape(dc.array(list(self.np_strides)), (2)).asTypeInt()
        self.dc_nullT      = dc.array(0)

        ## onnx conv example testcase
        self.onnx_dc_X      = dc.reshape(dc.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.]),(1,1,5,5))
        self.onnx_dc_X2     = dc.reshape(dc.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.]),(1,1,7,5))
        self.onnx_dc_W      = dc.reshape(dc.array([1.,1.,1.,1.,1.,1.,1.,1.,1.]),(1,1,3,3))
        self.onnx_npr_su    = np.array([12.,21.,27., 33., 24., 33., 54., 63., 72., 51., 63.,99.,108.,117., 81., 93.,144.,153.,162.,111., 72.,111.,117.,123., 84.])
        self.onnx_npr_vl    = np.array([54.,63.,72.,99.,108.,117.,144.,153.,162.])
        self.onnx_npr_vl_s2 = np.array([54.,72.,144.,162.,234.,252.])
        self.onnx_npr_sp_s2 = np.array([12.,27.,24.,63.,108.,81.,123.,198.,141.,112.,177.,124.])
        self.onnx_npr_ap_s2 = np.array([21.,33.,99.,117.,189.,207.,171.,183.])
        self.onnx_dc_BIGW   = dc.reshape(dc.array(list(np.ones(900))),(1,1,30,30))

#    def test_conv_nopad (self):
#        npr = signal.convolve2d(self.np_X, self.np_K)
#        dcr = dc.conv(self.dc_X,         # input
#                      self.dc_K,         # filter
#                      self.dc_B )          # bias
#                     "VALID",            # autopad
#                      self.dc_nullT,     # dilations
#                      1,                 # group
#                      self.dc_K.shape(), # kernel shape
#                      self.dc_nullT,     # pads
#                      self.dc_nullT     # strides
#                      )
#        npr and cr don't match - just a sanity test for now
#        np.testing.assert_allclose(npr, np.array(dcr.data()).astype(np.float32),rtol=1e-3, atol=1e-3)

# Ref: https://github.com/onnx/onnx/blob/master/onnx/backend/test/case/node/conv.py

# padding for same size
    def test_onnx_conv_same_upper (self):
        dcr            = dc.conv(self.onnx_dc_X, self.onnx_dc_W,  self.dc_B, "SAME_UPPER")
        np.testing.assert_allclose(self.onnx_npr_su.astype(np.float32), np.array(dcr.data()).astype(np.float32),rtol=1e-3, atol=1e-3)


# padding for same size
    def test_onnx_conv_same_lower (self):
        dcr            = dc.conv(self.onnx_dc_X, self.onnx_dc_W,  self.dc_B, "SAME_LOWER")
        np.testing.assert_allclose(self.onnx_npr_su.astype(np.float32), np.array(dcr.data()).astype(np.float32),rtol=1e-3, atol=1e-3)


# no padding
    def test_onnx_conv_valid (self):
        dcr            = dc.conv(self.onnx_dc_X, self.onnx_dc_W,  self.dc_B, "VALID")
        np.testing.assert_allclose(self.onnx_npr_vl.astype(np.float32), np.array(dcr.data()).astype(np.float32),rtol=1e-3, atol=1e-3)

# stride 2, no padding
    def test_onnx_conv_valid_s2 (self):
        dcr            = dc.conv(self.onnx_dc_X2, self.onnx_dc_W,  self.dc_B,
                                 "VALID",
                                 dc.vectorInt([]),
                                 1,
                                 dc.vectorInt([]),
                                 dc.vectorInt([]),
                                 dc.vectorInt([2,2]))
        np.testing.assert_allclose(self.onnx_npr_vl_s2.astype(np.float32), np.array(dcr.data()).astype(np.float32),rtol=1e-3, atol=1e-3)


# stride 2, explicit symmetrical padding
    def test_onnx_conv_ns_s2 (self):
        dcr            = dc.conv(self.onnx_dc_X2, self.onnx_dc_W,  self.dc_B,
                                 "NOTSET",
                                 dc.vectorInt([]),
                                 1,
                                 dc.vectorInt([]),
                                 dc.vectorInt([1,1,1,1]),
                                 dc.vectorInt([2,2]))
        np.testing.assert_allclose(self.onnx_npr_sp_s2.astype(np.float32), np.array(dcr.data()).astype(np.float32),rtol=1e-3, atol=1e-3)

# stride 2, explicit asymmetrical padding
    def test_onnx_conv_ns_ap_s2 (self):
        dcr            = dc.conv(self.onnx_dc_X2, self.onnx_dc_W,  self.dc_B,
                                 "NOTSET",
                                 dc.vectorInt([]),
                                 1,
                                 dc.vectorInt([]),
                                 dc.vectorInt([1,0,1,0]),
                                 dc.vectorInt([2,2]))
        np.testing.assert_allclose(self.onnx_npr_ap_s2.astype(np.float32), np.array(dcr.data()).astype(np.float32),rtol=1e-3, atol=1e-3)

# negative, kernel too big
    # def test_onnx_conv_same_upper (self):
    #     dcr = dc.conv(self.onnx_dc_X, self.onnx_dc_BIGW,  self.dc_B, "VALID")

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

