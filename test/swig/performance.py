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
DNNC_ROOT='/home/amd/speedygazelle/dnncompiler'
sys.path.append(os.path.abspath(DNNC_ROOT+os.path.sep+'swig'));

import dnnc as dc
import numpy as np
import time

class NPMatMul():
    def __init__(self, n):
        self.N = n
        self.np_a = np.random.randn(n*n).astype(np.float32)
        self.np_b = np.random.randn(n*n).astype(np.float32)

    def MatMul3D (self):
        np_a = np.reshape(self.np_a, (self.N, self.N))
        np_b = np.reshape(self.np_b, (self.N, self.N))

        npr = np.add(np_a, np_b);

class DCMatMul():
    def __init__(self, n):
        self.N = n
        self.dc_a = dc.random(n,n);
        self.dc_b = dc.random(n,n);

    def MatMul3D (self):
        dc_a = dc.reshape(self.dc_a, (self.N, self.N))
        dc_b = dc.reshape(self.dc_b, (self.N, self.N))

        dcr = dc.add(dc_a, dc_b);


if __name__ == '__main__':
    N=200

    for N in [20, 50, 100, 500, 1000, 2000, 5000, 10000]:
        nmpy = NPMatMul(N);
        start = time.time()
        nmpy.MatMul3D();
        np_time = time.time()-start

        start = time.time()
        dmat = DCMatMul(N);
        start = time.time()
        dmat.MatMul3D();
        dc_time = time.time()-start;


        print (N, np_time, dc_time, int(dc_time/np_time))
