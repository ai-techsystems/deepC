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

import deepC.dnnc as dc
import numpy as np
import unittest, random, math


def temp_softsign(x):
    return (x / (1 + np.abs(x)));

def temp_erf(x):
    y = np.vectorize(math.erf)(x).astype(np.float32)
    return y

class nnScalarOperatorsTest(unittest.TestCase):
    def setUp(self):
        self.random_number1 = random.randrange(20, 50, 3)
        self.random_number2 = random.randrange(200, 500, 1)
        self.random_number3 = random.randrange(10, 500, 2)
        # self.np_a = np.array(self.random_number1).astype(np.float32)
        # self.np_b = np.array(self.random_number2).astype(np.float32)
        # self.dc_a = dc.array([self.random_number1])
        # self.dc_b = dc.array([self.random_number2])
        self.np_a = self.random_number1
        self.np_b = self.random_number2
        self.dc_a = self.random_number1
        self.dc_b = self.random_number2

    def test_nnScalar_asin (self):
        np.testing.assert_allclose(np.arcsin(1), dc.asin(1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.arcsin(0), dc.asin(0), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.arcsin(-1), dc.asin(-1), rtol=1e-3, atol=1e-3)

    def test_nnScalar_acos (self):
        np.testing.assert_allclose(np.arccos(1), dc.acos(1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.arccos(0), dc.acos(0), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.arccos(-1), dc.acos(-1), rtol=1e-3, atol=1e-3)

    def test_nnScalar_atan (self):
        np.testing.assert_allclose(np.arctan(self.random_number1), dc.atan(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.arctan(self.random_number2), dc.atan(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.arctan(self.random_number3), dc.atan(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_asinh (self):
        np.testing.assert_allclose(np.arcsinh(self.random_number1), dc.asinh(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.arcsinh(self.random_number2), dc.asinh(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.arcsinh(self.random_number3), dc.asinh(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_acosh (self):
        np.testing.assert_allclose(np.arccosh(self.random_number1), dc.acosh(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.arccosh(self.random_number2), dc.acosh(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.arccosh(self.random_number3), dc.acosh(self.random_number3), rtol=1e-3, atol=1e-3)

    # def test_nnScalar_atanh (self):
    #     np.testing.assert_allclose(np.arctanh(self.random_number1), dc.atanh(self.random_number1), rtol=1e-3, atol=1e-3)
    #     np.testing.assert_allclose(np.arctanh(self.random_number2), dc.atanh(self.random_number2), rtol=1e-3, atol=1e-3)
    #     np.testing.assert_allclose(np.arctanh(self.random_number3), dc.atanh(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_sin (self):
        np.testing.assert_allclose(np.sin(self.random_number1), dc.sin(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.sin(self.random_number2), dc.sin(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.sin(self.random_number3), dc.sin(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_cos (self):
        np.testing.assert_allclose(np.cos(self.random_number1), dc.cos(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.cos(self.random_number2), dc.cos(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.cos(self.random_number3), dc.cos(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_tan (self):
        np.testing.assert_allclose(np.tan(self.random_number1), dc.tan(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.tan(self.random_number2), dc.tan(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.tan(self.random_number3), dc.tan(self.random_number3), rtol=1e-3, atol=1e-3)

    # def test_nnScalar_sinh (self):
    #     np.testing.assert_allclose(np.sinh(self.random_number1), dc.sinh(self.random_number1), rtol=1e-3, atol=1e-3)
    #     np.testing.assert_allclose(np.sinh(self.random_number2), dc.sinh(self.random_number2), rtol=1e-3, atol=1e-3)
    #     np.testing.assert_allclose(np.sinh(self.random_number3), dc.sinh(self.random_number3), rtol=1e-3, atol=1e-3)

    # def test_nnScalar_cosh (self):
    #     np.testing.assert_allclose(np.cosh(self.random_number1), dc.cosh(self.random_number1), rtol=1e-3, atol=1e-3)
    #     np.testing.assert_allclose(np.cosh(self.random_number2), dc.cosh(self.random_number2), rtol=1e-3, atol=1e-3)
    #     np.testing.assert_allclose(np.cosh(self.random_number3), dc.cosh(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_tanh (self):
        np.testing.assert_allclose(np.tanh(self.random_number1), dc.tanh(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.tanh(self.random_number2), dc.tanh(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.tanh(self.random_number3), dc.tanh(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_erf (self):
        np.testing.assert_allclose(temp_erf(self.random_number1), dc.erf(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(temp_erf(self.random_number2), dc.erf(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(temp_erf(self.random_number3), dc.erf(self.random_number3), rtol=1e-3, atol=1e-3)

    # def test_nnScalar_exp (self):
    #     np.testing.assert_allclose(np.exp(self.random_number1), dc.exp(self.random_number1), rtol=1e-3, atol=1e-3)
    #     np.testing.assert_allclose(np.exp(self.random_number2), dc.exp(self.random_number2), rtol=1e-3, atol=1e-3)
    #     np.testing.assert_allclose(np.exp(self.random_number3), dc.exp(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_log (self):
        np.testing.assert_allclose(np.log(self.random_number1), dc.log(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.log(self.random_number2), dc.log(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.log(self.random_number3), dc.log(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_logical_not (self):
        np.testing.assert_allclose(np.logical_not(self.random_number1), dc.logical_not(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.logical_not(self.random_number2), dc.logical_not(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.logical_not(self.random_number3), dc.logical_not(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_sign (self):
        np.testing.assert_allclose(np.sign(self.random_number1), dc.sign(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.sign(self.random_number2), dc.sign(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.sign(self.random_number3), dc.sign(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_softsign (self):
        np.testing.assert_allclose(temp_softsign(self.random_number1), dc.softsign(self.random_number1), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(temp_softsign(self.random_number2), dc.softsign(self.random_number2), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(temp_softsign(self.random_number3), dc.softsign(self.random_number3), rtol=1e-3, atol=1e-3)

    def test_nnScalar_max (self):
        npr = np.maximum(self.np_a, self.np_b)
        dcr = dc.max([self.dc_a,self.dc_b])
        np.testing.assert_allclose(npr, np.array(dcr).astype(np.float32),rtol=1e-3, atol=1e-3)

    def test_nnScalar_min (self):
        npr = np.minimum(self.np_a, self.np_b)
        dcr = dc.min([self.dc_a,self.dc_b])
        np.testing.assert_allclose(npr, np.array(dcr).astype(np.float32),rtol=1e-3, atol=1e-3)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

