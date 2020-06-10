#!/usr/bin/env python3

import os, sys, glob

import subprocess
import unittest
import deepC

# This test runs compiler as a user would run on command line.
class mnistTest(unittest.TestCase):


    def setUp(self):
        self.debug=False;

        test_dir = os.path.join("compiler", "mnist")
        for wcard in [ '*bias', '*weight', '*out', '*cpp', '*exe']:
            for filename in glob.glob(os.path.join(test_dir, wcard)):
                os.remove(filename)
        compile_scr = os.path.join(os.path.dirname(deepC.__file__), "compiler", "onnx2exe.py")
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
        test_env = os.environ.copy();
        if ( "PYTHONPATH" not in test_env ) :
            test_env["PYTHONPATH"]=os.path.abspath(os.path.join(os.getcwd(),'../../'))
        else:
            test_env["PYTHONPATH"]+=":"+os.path.abspath(os.path.join(os.getcwd(),'../../'))

        for cmd in self.commands:
            test_proc=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, env=test_env)
            try:
                outs, errs = test_proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                test_proc.kill()
                outs, errs = test_proc.communicate()

            if ( self.debug ):
                print(cmd)
                print(outs.decode())


    def tearDown(self):
        return "test finished"

if __name__ == '__main__':
    unittest.main()

