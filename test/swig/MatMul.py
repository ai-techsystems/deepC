import os,sys
DNNC_ROOT='/home/amd/speedygazelle/dnncompiler'
sys.path.append(os.path.abspath(DNNC_ROOT+os.path.sep+'swig'));

import dnnc as dc
import numpy as np
import unittest

class MatMulTest(unittest.TestCase):
    def rowMajor2ColumnMajor2D(self,mat2D):
        if(isinstance(mat2D, np.ndarray)):
            d0 = mat2D.shape[0]
            d1 = mat2D.shape[1]
        else:
            d0 = mat2D.shape()[0]
            d1 = mat2D.shape()[1]
        len = d0*d1
        npl = [float(i) for i in range(len)];
        m=0
        for j in range(d1):
            for i in range(d0):
                if(isinstance(mat2D, np.ndarray)):
                    npl[m] = float(mat2D[i,j])
                else:
                    npl[m] = float(mat2D(i,j))
                m=m+1
        npl_vec = dc.fvec(npl)
        dc2D = dc.array(d0,d1)
        dc2D.load(npl_vec)
        return dc2D

    def rowMajor2ColumnMajor3D(self,tensor3D):
        if(isinstance(tensor3D, np.ndarray)):
            d0 = tensor3D.shape[0]
            d1 = tensor3D.shape[1]
            d2 = tensor3D.shape[2]
        else:
            d0 = tensor3D.shape()[0]
            d1 = tensor3D.shape()[1]
            d2 = tensor3D.shape()[2]
        len = d0*d1*d2
        npl = [float(i) for i in range(len)];
        m=0
        for i in range(d0):
            for k in range(d2):
                for j in range(d1):
                    if(isinstance(tensor3D, np.ndarray)):
                        npl[m] = float(tensor3D[i,j,k])
                    else:
                        npl[m] = float(tensor3D(i,j,k))
                    m=m+1
        npl_vec = dc.fvec(npl)
        dc3D = dc.array(d1,d2,d0)
        dc3D.load(npl_vec)
        return dc3D

    def __init__(self):
        self.len = 12
        self.np_a = np.random.randn(self.len).astype(np.float32)
        self.np_b = np.random.randn(self.len).astype(np.float32)
        #self.np_a = np.arange(self.len).astype(np.float32)
        #self.np_b = np.arange(self.len).astype(np.float32)
        self.dc_a = dc.array(list(self.np_a));
        self.dc_b = dc.array(list(self.np_b));

    def MatMul1D (self):
        npr = np.matmul(self.np_a, self.np_b)
        dcr = dc.matmul(self.dc_a, self.dc_b)
        np.testing.assert_allclose(npr, np.array(dcr.data()[0]).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def MatMul2D (self):
        np_a = self.np_a.reshape(3,4)
        np_b = self.np_b.reshape(4,3)
        dc_a = dc.reshape(self.dc_a, (3,4));
        dc_b = dc.reshape(self.dc_b, (4,3));
        npr = np.matmul(np_a, np_b);
        dcr = dc.matmul(dc_a, dc_b);
        np.testing.assert_allclose(npr.flatten(order='F'), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

    def MatMul3D (self):
        np_a = self.np_a.reshape(2,2,3)
        np_b = self.np_b.reshape(2,3,2);
        dc_a = self.rowMajor2ColumnMajor3D(np_a);
        dc_b = self.rowMajor2ColumnMajor3D(np_b)

        npr = np.matmul(np_a, np_b);
        dcr = dc.matmul(dc_a, dc_b);
        dcr_list = [i for i in range(8)];
        dcr_data = dcr.data();
        for i,j in zip(range(8), [0, 4, 1, 5, 2, 6, 3, 7]):
            dcr_list[i] = dcr_data[j]
        dcr = dc.array(dcr_list)
        np.testing.assert_allclose(npr.flatten(order='F'), np.array(dcr.data()).astype(np.float32),
                rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
    m = MatMulTest()
    m.MatMul1D()
    m.MatMul2D()
    m.MatMul3D()
    unittest.main()
