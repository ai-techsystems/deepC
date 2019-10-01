import os,sys
DNNC_ROOT=os.path.abspath(os.path.dirname(__file__))
sys.path.append(DNNC_ROOT+os.path.sep+'swig')
sys.path.append(DNNC_ROOT+os.path.sep+'python')

from swig import dnnc as dc
from python import read_onnx
