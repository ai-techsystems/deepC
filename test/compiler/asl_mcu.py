#!/usr/bin/env python3

import os, sys

from subprocess import PIPE, run
import unittest
import deepC

# This test runs compiler as a user would run on command line.
class asl_mcuTest(unittest.TestCase):
    def setUp(self):
        compile_scr = os.path.join(os.path.dirname(deepC.__file__), "scripts", "onnx2exe.py")
        test_dir = os.path.join("compiler", "asl_mcu")
        gesture_data_file = os.path.join(test_dir, "gesture.data")
        self.commands = [
            # delete last generated file.
            "rm -f " + os.path.join(test_dir, "asl_model.exe"),
            # compile onnx into exe
            "python3 " + compile_scr + " " + os.path.join(test_dir, "asl_model.onnx"),
            # check executable file.
            "ls -l " + os.path.join(test_dir, "asl_model.exe"),
            # run it with relative path.
            os.path.join(test_dir, "asl_model.exe") + " " + gesture_data_file,
            # run it in bundle dir (current dir)
            "cd " + test_dir + "; ./asl_model.exe gesture.data",
            # run it with absolute path
            os.path.join(os.getcwd(), test_dir, "asl_model.exe") + " " + gesture_data_file
            ]

    def test_runModel(self):
        for cmd in self.commands:
            run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True, check=True)

    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

