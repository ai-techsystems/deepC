import os, sys, fnmatch

import unittest
import read_onnx

class unitOperatorsTest(unittest.TestCase):

    def find(self, pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    def test_readAllOnnxFiles(self):
        # mute stdout
        sys_stdout = sys.stdout
        f = open(os.devnull, 'w')
        sys.stdout = f

        onnx_files = self.find('*.onnx', '.')
        for onnx_file in onnx_files:
            parser = read_onnx.pbReader()
            parser.main(onnx_file)

        # unmute stdout
        sys.stdout = sys_stdout

        print("read %d files." %len(onnx_files))
        assert(len(onnx_files)==131)
