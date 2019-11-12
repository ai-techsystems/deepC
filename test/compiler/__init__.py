import os, sys, importlib
import unittest

# def load_tests(loader, tests, pattern):
def load_tests(loader, tests):
    # suite = unittest.TestSuite();

    test_file = "compiler/passingTests.txt";
    if ( os.path.isfile(test_file) == False ):
        print("no test file in ", os.getcwd()+"/compiler");
        # return suite;
        return

    test_fp = open(test_file, "r");
    #print("opened file", test_file)

    for test in test_fp.readlines():

        module_name = test.strip().split(".")[0]
        class_name = module_name + "Test"
        module = importlib.import_module("."+module_name, package="compiler")
        class_ = getattr(module, class_name)

        tests.append(loader.loadTestsFromTestCase(class_))

    test_fp.close()
    return

    # return suite;

def load_test(loader, test, tests):

    test_file = "compiler/"+test
    if ( os.path.isfile(test_file) == False ):
        print("no test file in ", os.getcwd()+"/compiler");
        # return suite;
        return

    print("running test", test.strip())
    module_name = test.strip().split(".")[0]
    class_name = module_name + "Test"
    module = importlib.import_module("."+module_name, package="compiler")
    class_ = getattr(module, class_name)

    tests.append(loader.loadTestsFromTestCase(class_))

    return
