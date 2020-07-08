import os, sys, fnmatch

import unittest
import deepC.compiler.read_onnx as read_onnx
import deepC.dnnc as dnnc

class unitOperatorsTest(unittest.TestCase):

    def find(self, pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    def test_readAllOnnxFiles(self):
        debug = False;
        if debug == False :
            # mute stdout
            sys_stdout = sys.stdout
            f = open(os.devnull, 'w')
            sys.stdout = f

        onnx_files = self.find('*.onnx', 'parser')
        for onnx_file in onnx_files:
            if ( debug ):
                print("testing ", onnx_file, flush=True)
            cpp_file = os.path.splitext(os.path.basename(onnx_file))[0]+'.cpp'
            bundle_dir = os.path.dirname(onnx_file);
            parser = read_onnx.pbReader()
            dc_graph = parser.main(onnx_file)
            cppCode = dnnc.cppCodeGen(dc_graph, bundle_dir, cpp_file);
            cppCode.write();
            dc_graph.destroy();

        # remove generated cpp files.
        for cpp_file in self.find('*.cpp', 'parser'):
            os.remove(cpp_file)

        # unmute stdout
        if debug == False :
            sys.stdout = sys_stdout

        print("read %d files." %len(onnx_files))
        assert(len(onnx_files)==130)
