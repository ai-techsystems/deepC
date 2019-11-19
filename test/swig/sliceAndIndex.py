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

		self.np_1D = np.arange(12).astype(np.int)
		self.dc_1D = dc.arange(12).asTypeInt()
		self.np_2D = np.arange(12).reshape(4,3).astype(np.int)
		self.dc_2D = dc.arange(12).reshape(4,3).asTypeInt()
		self.np_3D = np.arange(48).reshape(4,3,4).astype(np.int)
		self.dc_3D = dc.arange(48).reshape(4,3,4).asTypeInt()
		self.np_4D = np.arange(96).reshape(4,3,2,4).astype(np.int)
		self.dc_4D = dc.arange(96).reshape(4,3,2,4).asTypeInt()

	def test_getitem(self):

		# Indexing
		np.testing.assert_array_equal(self.np_1D[2], self.dc_1D[2])
		np.testing.assert_array_equal(self.np_2D[2], (self.dc_2D[2]).numpy())
		np.testing.assert_array_equal(self.np_3D[2,1], (self.dc_3D[2,1]).numpy())
		np.testing.assert_array_equal(self.np_4D[1,2], (self.dc_4D[1,2]).numpy())
		# special case as dnnc prints python datatype, but numpy prints numpy datatype
		np.testing.assert_array_equal(int(self.np_2D[2,1]), int(self.dc_2D[2,1]))

		# Slicing
		np.testing.assert_array_equal(self.np_1D[2:3], (self.dc_1D[2:3]).numpy())
		np.testing.assert_array_equal(self.np_1D[2:], (self.dc_1D[2:]).numpy())
		np.testing.assert_array_equal(self.np_1D[::3], (self.dc_1D[::3]).numpy())
		np.testing.assert_array_equal(self.np_2D[2:3,:], (self.dc_2D[2:3,:]).numpy())
		np.testing.assert_array_equal(self.np_2D[2:,1:], (self.dc_2D[2:,1:]).numpy())
		np.testing.assert_array_equal(self.np_2D[:,::3], (self.dc_2D[:,::3]).numpy())
		np.testing.assert_array_equal(self.np_3D[2:3,:,1:2], (self.dc_3D[2:3,:,1:2]).numpy())
		np.testing.assert_array_equal(self.np_3D[2:,1:,1:2], (self.dc_3D[2:,1:,1:2]).numpy())
		np.testing.assert_array_equal(self.np_3D[:,::3,1:2], (self.dc_3D[:,::3,1:2]).numpy())
		np.testing.assert_array_equal(self.np_4D[2:3,:,:1:2,::2], (self.dc_4D[2:3,:,:1:2,::2]).numpy())
		np.testing.assert_array_equal(self.np_4D[2:,1:,:1:2,::2], (self.dc_4D[2:,1:,:1:2,::2]).numpy())
		np.testing.assert_array_equal(self.np_4D[:,::3,:1:2], (self.dc_4D[:,::3,:1:2]).numpy())

		# Slicing with Indexing
		np.testing.assert_array_equal(self.np_2D[2:,1], (self.dc_2D[2:,1]).numpy())
		np.testing.assert_array_equal(self.np_2D[2,::3], (self.dc_2D[2,::3]).numpy())
		np.testing.assert_array_equal(self.np_3D[2:,1,::2], (self.dc_3D[2:,1,::2]).numpy())

		## B U G     numpy has shape with null tensors
		# np.testing.assert_array_equal(self.np_3D[2,::3,3:1], (self.dc_3D[2,::3,3:1]).numpy())
		# np.testing.assert_array_equal(self.np_4D[2:,1,0,2:1:3], (self.dc_4D[2:,1,0,2:1:3]).numpy())
		np.testing.assert_array_equal(self.np_4D[2,::3,0:-1:3], (self.dc_4D[2,::3,0:-1:3]).numpy())

		# Ellipsis with Slicing
		np.testing.assert_array_equal(self.np_2D[...,::-1], (self.dc_2D[...,::-1]).numpy())
		np.testing.assert_array_equal(self.np_2D[1:,...], (self.dc_2D[1:,...]).numpy())
		np.testing.assert_array_equal(self.np_3D[...,::-1], (self.dc_3D[...,::-1]).numpy())
		np.testing.assert_array_equal(self.np_3D[1:,...], (self.dc_3D[1:,...]).numpy())
		np.testing.assert_array_equal(self.np_4D[...,::-1], (self.dc_4D[...,::-1]).numpy())
		np.testing.assert_array_equal(self.np_4D[1:,...], (self.dc_4D[1:,...]).numpy())

		# Ellipsis with Indexing
		np.testing.assert_array_equal(self.np_2D[...,1], (self.dc_2D[...,1]).numpy())
		np.testing.assert_array_equal(self.np_2D[2,...], (self.dc_2D[2,...]).numpy())
		np.testing.assert_array_equal(self.np_3D[...,1], (self.dc_3D[...,1]).numpy())
		np.testing.assert_array_equal(self.np_3D[2,...], (self.dc_3D[2,...]).numpy())
		np.testing.assert_array_equal(self.np_4D[...,1], (self.dc_4D[...,1]).numpy())
		np.testing.assert_array_equal(self.np_4D[2,...], (self.dc_4D[2,...]).numpy())

	def test_setitem(self):

		# Indexing
		self.np_2D[2] = 200
		self.dc_2D[2] = 200
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[1,2] = 500
		self.dc_2D[1,2] = 500
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_3D[2,1] = 200
		self.dc_3D[2,1] = 200
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_3D[1,2] = 500
		self.dc_3D[1,2] = 500
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_4D[2,1] = 200
		self.dc_4D[2,1] = 200
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())
		self.np_4D[1,2] = 500
		self.dc_4D[1,2] = 500
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())

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
		self.np_2D[:,::-2] = 45
		self.dc_2D[:,::-2] = 45
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_3D[2:3,:,1] = 500
		self.dc_3D[2:3,:,1] = 500
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_3D[2:,1:] = 30
		self.dc_3D[2:,1:] = 30
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_3D[:,::3] = 25
		self.dc_3D[:,::3] = 25
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_3D[:,::3] = 45
		self.dc_3D[:,::3] = 45
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_4D[2:3,:,1] = 500
		self.dc_4D[2:3,:,1] = 500
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())
		self.np_4D[2:,1:] = 30
		self.dc_4D[2:,1:] = 30
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())
		self.np_4D[:,::3] = 25
		self.dc_4D[:,::3] = 25
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())
		self.np_4D[:,0:1:-1] = 45
		self.dc_4D[:,0:1:-1] = 45
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())

		# Slicing with Indexing
		self.np_2D[2:,1] = 65
		self.dc_2D[2:,1] = 65
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[2,::3] = 75
		self.dc_2D[2,::3] = 75
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_3D[2:,1] = 65
		self.dc_3D[2:,1] = 65
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_3D[2,::3] = 75
		self.dc_3D[2,::3] = 75
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_4D[2:,1] = 65
		self.dc_4D[2:,1] = 65
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())
		self.np_4D[2,::3] = 75
		self.dc_4D[2,::3] = 75
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())

		# Ellipsis with Slicing
		self.np_2D[...,::-1] = 62
		self.dc_2D[...,::-1] = 62
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[1:,...] = 73
		self.dc_2D[1:,...] = 73
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_3D[...,::-1] = 62
		self.dc_3D[...,::-1] = 62
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_3D[1:,...] = 73
		self.dc_3D[1:,...] = 73
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_4D[...,::-1] = 62
		self.dc_4D[...,::-1] = 62
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())
		self.np_4D[1:,...] = 73
		self.dc_4D[1:,...] = 73
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())

		# Ellipsis with Indexing
		self.np_2D[...,1] = 63
		self.dc_2D[...,1] = 63
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_2D[2,...] = 71
		self.dc_2D[2,...] = 71
		np.testing.assert_array_equal(self.np_2D, (self.dc_2D).numpy())
		self.np_3D[...,1] = 63
		self.dc_3D[...,1] = 63
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_3D[2,...] = 71
		self.dc_3D[2,...] = 71
		np.testing.assert_array_equal(self.np_3D, (self.dc_3D).numpy())
		self.np_4D[...,1] = 63
		self.dc_4D[...,1] = 63
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())
		self.np_4D[2,...] = 71
		self.dc_4D[2,...] = 71
		np.testing.assert_array_equal(self.np_4D, (self.dc_4D).numpy())


	def tearDown(self):
		return "test finished"


if __name__ == '__main__':

	unittest.main()