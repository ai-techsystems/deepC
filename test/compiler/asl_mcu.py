#!/usr/bin/env python3

import os, sys, glob

import subprocess
import unittest
import deepC

# This test runs compiler as a user would run on command line.
class asl_mcuTest(unittest.TestCase):


    def setUp(self):
        self.debug=False;

        self.test_dir = os.path.join("compiler", "asl_mcu")
        for wcard in [ '*bias', '*weight', '*out', '*cpp', '*exe']:
            for filename in glob.glob(os.path.join(self.test_dir, wcard)):
                os.remove(filename)
        compile_scr = os.path.join(os.path.dirname(deepC.__file__), "compiler", "onnx2exe.py")
        gesture_data_file = os.path.join(self.test_dir, "gesture.data")

        self.commands = [
            # delete last generated file.
            "rm -f " + os.path.join(self.test_dir, "asl_model.exe"),
            # compile onnx into exe
            sys.executable + " " + compile_scr + " " + os.path.join(self.test_dir, "asl_model.onnx"),
            # check executable file.
            "ls -l " + os.path.join(self.test_dir, "asl_model.exe"),
            # run it with relative path.
            os.path.join(self.test_dir, "asl_model.exe") + " " + gesture_data_file,
            # run it in bundle dir (current dir)
            "cd " + self.test_dir + " && ./asl_model.exe gesture.data",
            # run it with absolute path
            os.path.join(os.getcwd(), self.test_dir, "asl_model.exe") + " " + gesture_data_file
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

