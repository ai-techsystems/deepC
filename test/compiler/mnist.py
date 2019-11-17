#!/usr/bin/env python3

import os, sys

from subprocess import PIPE, run
import unittest
import deepC

# This test runs compiler as a user would run on command line.
class mnistTest(unittest.TestCase):
    def setUp(self):
        compile_scr = os.path.join(os.path.dirname(deepC.__file__), "scripts", "onnx2exe.py")
        test_dir = os.path.join("compiler", "mnist")
        image_file = os.path.join(test_dir, "image.txt")
        self.commands = [
            # delete last generated file.
            "rm -f " + os.path.join(test_dir, "mnist.exe"),
            # compile onnx into exe
            "python3 " + compile_scr + " " + os.path.join(test_dir, "mnist.onnx"),
            # check executable file.
            "ls -l " + os.path.join(test_dir, "mnist.exe"),
            # run it with relative path.
            os.path.join(test_dir, "mnist.exe") + " " + image_file,
            # run it in bundle dir (current dir)
            "cd " + test_dir + "; ./mnist.exe image.txt",
            # run it with absolute path
            os.path.join(os.getcwd(), test_dir, "mnist.exe") + " " + image_file
            ]

    def test_runModel(self):
        for cmd in self.commands:
            run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True, check=True)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

