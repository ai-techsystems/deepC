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

import numpy as np
import tensorflow as tf
import dnnc as dc

import time

class NPMatMul():
    def __init__(self, n):
        self.N = n
        self.np_a = np.random.randn(n*n).astype(np.float32)
        self.np_b = np.random.randn(n*n).astype(np.float32)

    def MatMul3D (self):
        np_a = np.reshape(self.np_a, (self.N, self.N))
        np_b = np.reshape(self.np_b, (self.N, self.N))

        npr = np.matmul(np_a, np_b);

class DCMatMul():
    def __init__(self, n):
        self.N = n
        self.dc_a = dc.random(n,n);
        self.dc_b = dc.random(n,n);

    def MatMul3D (self):
        dc_a = dc.reshape(self.dc_a, (self.N, self.N))
        dc_b = dc.reshape(self.dc_b, (self.N, self.N))

        dcr = dc.matmul(dc_a, dc_b);

class TFMatMul():
    def __init__(self, n):
        tf.enable_eager_execution()
        self.N = n
        self.tf_a = tf.get_variable("tf_a", initializer=tf.random_uniform([n, n], dtype=tf.float32))
        self.tf_b = tf.get_variable("tf_b", initializer=tf.random_uniform([n, n], dtype=tf.float32))

    def MatMul3D (self):
        tfr = tf.matmul(self.tf_a, self.tf_b)

def fmt(n):
    return "{0:.2g}".format(n)

if __name__ == '__main__':
    N=200

    print ("Matrix(NxN) DC NP TF DC/NP DC/TF")
    for N in [20, 50, 200, 500, 2000, 5000]:
        nmpy = NPMatMul(N);
        start = time.time()
        nmpy.MatMul3D();
        np_time = time.time()-start

        tfmm = TFMatMul(N);
        start = time.time()
        tfmm.MatMul3D();
        tf_time = time.time()-start;

        dcmm = DCMatMul(N);
        start = time.time()
        dcmm.MatMul3D();
        dc_time = time.time()-start;

        print (N, fmt(dc_time), fmt(np_time), fmt(tf_time), fmt(dc_time/np_time), fmt(dc_time/tf_time))

