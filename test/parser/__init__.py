import os, sys, importlib
import unittest

DNNC_ROOT=os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..'+os.path.sep+'..')
sys.path.append(DNNC_ROOT)
sys.path.append(DNNC_ROOT+os.path.sep+'swig')
sys.path.append(DNNC_ROOT+os.path.sep+'python')

# def load_tests(loader, tests, pattern):
def load_tests(loader, tests):
    # suite = unittest.TestSuite();

    test_file = "parser/passingTests.txt";
    if ( os.path.isfile(test_file) == False ):
        print("no test file in ", os.getcwd()+"/parser");
        # return suite;
        return

    test_fp = open(test_file, "r");
    #print("opened file", test_file)

    for test in test_fp.readlines():

        module_name = test.strip().split(".")[0]
        class_name = module_name + "Test"
        module = importlib.import_module("."+module_name, package="parser")
        class_ = getattr(module, class_name)

        tests.append(loader.loadTestsFromTestCase(class_))

    test_fp.close()
    return

    # return suite;

def load_test(loader, test, tests):

    test_file = "parser/"+test
    if ( os.path.isfile(test_file) == False ):
        print("no test file in ", os.getcwd()+"/parser");
        # return suite;
        return

    print("running test", test.strip())
    module_name = test.strip().split(".")[0]
    class_name = module_name + "Test"
    module = importlib.import_module("."+module_name, package="parser")
    class_ = getattr(module, class_name)

    tests.append(loader.loadTestsFromTestCase(class_))

    return
