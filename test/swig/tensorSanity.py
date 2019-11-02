# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for divitional information
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

import deepC.dnnc as dc
import numpy as np
import unittest

class tensorSanityTest(unittest.TestCase):
    def setUp(self):
        return;

    # how to create tensors in different ways
    def test_create(self):

        # null tensor test
        a=dc.array(0);
        assert a.isnull() == True
        assert a.empty() == True

        # test assignment is shallow copy of memory
        b=a
        assert a.sameas(b) == True
        assert a.identifier() == b.identifier()

        # tensor without initiliaztion
        a=dc.array(2,3,4,5);
        assert a.length() == 120

        # tensor random initiliaztion
        a=dc.random(2,3,4,5);
        assert a.length() == 120

        # tensor without initiliaztion
        a=dc.empty(2,3,4,5);
        assert a.length() == 120

        # zero tensor
        a=dc.zeros(2,3,4,5);
        assert np.array(list(a.data())).sum().astype(np.int) == 0

        # one tensor
        a=dc.ones(2,3,4,5);
        assert np.array(list(a.data())).sum().astype(np.int) == 120

        # tensor from python list
        l1D=[1,3,5]
        a=dc.array(l1D).asTypeInt()
        np.testing.assert_equal(np.array(l1D), np.array(list(a.data())))

        # tensor from python list of lists
        l2D=[[1,3,5],[2,4,6]]
        a=dc.array(l2D).asTypeInt()
        assert a.rank() == 2
        assert a.shape() == (2, 3)
        np.testing.assert_equal(np.array(l2D).flatten(), \
                np.array(list(a.data())))

        # copy tensor
        b=a.copy()
        assert a.sameas(b) == False
        assert a.identifier() != b.identifier()

        # arange
        a=dc.arange(10)
        assert a.length() == 10

        # add start and step
        a=dc.arange(10, 5, 3).asTypeInt()
        assert a.data() == (5, 8)

        # swap start and stop.
        a=dc.arange(5, 10, 3).asTypeInt()
        assert a.data() == (5, 8)

    # test data loading and index
    def test_data(self):

        # confirm type as class tuple.
        a=dc.zeros(2,3).asTypeInt()
        adata = a.data()
        assert type(adata) == type((1,))

        # load new data
        new_data_list = [10,11,12,13,14,15]
        a.load(dc.vectorInt(new_data_list))
        assert a[0] == 10

        # load one element with flat index
        a[0] = 777
        assert a[0] == 777

        # reshape, fetch and load with multi indices
        a=dc.arange(12).asTypeInt()
        a.reshape(dc.vectorSizeT([2,2,3]))
        assert a[0,1,1] == 4

        a[1,1,1] = 200
        assert a[1,1,1] == 200

        # negative test
        try :
            # This throws ValueError
            print(a[0,0,9,9,9])
        except ValueError as e:
            assert e


    # test data types
    def test_dtypes(self):
        a=dc.random(2,3)
        assert a.dtype() == 'float'

        # transform datatype to int.
        aint = a.asTypeInt()
        assert aint.dtype() == 'int32_t'

        # transform datatype to double.
        adbl = a.asTypeDouble();
        assert adbl.dtype() == 'double'

        # transform datatype to double.
        abool = a.asTypeBool()
        assert abool.dtype() == 'bool'

    # test shapes
    def test_shapes(self):

        # test shape tuple
        shape1=dc.vectorSizeT([2,3,4,5])
        shape2=dc.vectorSizeT([5,4,3,2])
        a=dc.random(2,3,4,5).asTypeInt()
        assert a.rank() == 4
        assert a.shape() == (2, 3, 4, 5)

        # reshape to new dimensions
        a.reshape(shape2)
        assert a.shape() == (5, 4, 3, 2)

        # return a new tensor with flattened dimensions.
        b=a.flatten()
        assert a.shape() == (5, 4, 3, 2)
        assert b.shape() == (120,)

        # flatten the same tensor
        a.flatteninplace()
        assert a.shape() == (120,)
        shape3=dc.vectorSizeT([8,15,1,1])

        # new shape
        a.reshape(shape3)

        # confirm new shape
        assert a.shape() == (8,15,1,1)

        # dnnc method to reshape.
        a=dc.random(2,3,4,5)
        dc.reshape(a,(8,15,1,1))
        assert a.shape() == (8,15,1,1)
        dc.reshape(a,(120,))
        assert a.shape() == (120,)


    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

