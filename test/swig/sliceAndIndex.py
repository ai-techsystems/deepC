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

class sliceAndIndexTest(unittest.TestCase):

	def setUp(self):

		self.np_2D = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]]).astype(np.float)
		self.dc_2D = dc.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]]).asTypeFloat()

	def test_getitem(self):

		# Indexing
		np.testing.assert_array_equal(self.np_2D[2], (self.dc_2D[2]).numpy())
		# special case as dnnc prints python datatype, but numpy prints numpy datatype
		np.testing.assert_array_equal(int(self.np_2D[2,1]), int(self.dc_2D[2,1]))

		# Slicing
		np.testing.assert_array_equal(self.np_2D[2:3,:], (self.dc_2D[2:3,:]).numpy())
		np.testing.assert_array_equal(self.np_2D[2:,1:], (self.dc_2D[2:,1:]).numpy())
		np.testing.assert_array_equal(self.np_2D[:,::3], (self.dc_2D[:,::3]).numpy())

		# Slicing with Indexing
		np.testing.assert_array_equal(self.np_2D[2:,1], (self.dc_2D[2:,1]).numpy())
		np.testing.assert_array_equal(self.np_2D[2,::3], (self.dc_2D[2,::3]).numpy())

		# Ellipsis with Slicing
		np.testing.assert_array_equal(self.np_2D[...,::-1], (self.dc_2D[...,::-1]).numpy())
		np.testing.assert_array_equal(self.np_2D[1:,...], (self.dc_2D[1:,...]).numpy())

		# Ellipsis with Indexing
		np.testing.assert_array_equal(self.np_2D[...,1], (self.dc_2D[...,1]).numpy())
		np.testing.assert_array_equal(self.np_2D[2,...], (self.dc_2D[2,...]).numpy())

	def test_setitem(self):

		# Indexing
		self.np_2D[2] = 200
		self.dc_2D[2] = 200
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[1,2] = 500
		self.dc_2D[1,2] = 500
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())

		# Slicing
		self.np_2D[2:3,:] = [500, 200, 30]
		self.dc_2D[2:3,:] = [500, 200, 30]
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[2:,1:] = [50, 30]
		self.dc_2D[2:,1:] = [50, 30]
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[:,::3] = 25
		self.dc_2D[:,::3] = 25
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[:,::3] = 45
		self.dc_2D[:,::3] = 45
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())

		# Slicing with Indexing
		self.np_2D[2:,1] = 65
		self.dc_2D[2:,1] = 65
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[2,::3] = 75
		self.dc_2D[2,::3] = 75
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())

		# Ellipsis with Slicing
		self.np_2D[...,::-1] = 62
		self.dc_2D[...,::-1] = 62
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[1:,...] = 73
		self.dc_2D[1:,...] = 73
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())

		# Ellipsis with Indexing
		self.np_2D[...,1] = 63
		self.dc_2D[...,1] = 63
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[2,...] = 71
		self.dc_2D[2,...] = 71
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())


	def tearDown(self):
		return "test finished"


if __name__ == '__main__':

	unittest.main()

