# HOW TO USE THIS FILE TO BUILD python extension package
#   command: python3 setup.py build_ext --inplace
#
## TODO: remove -g flag from compilation.

import os

os.environ["DISTUTILS_DEBUG"]="1"
os.environ["Py_DEBUG"]="0"
os.environ["CC"] = "clang++-8"
os.environ["CXX"] = "clang++-8"
os.environ["LINKCC"] = "clang++-8"
os.environ["LDSHARED"] = "clang++-8 -shared"
os.environ["OPT"] = "-fwrapv -O2 -Wall"
os.environ["CFLAGS"] = "-Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -fstack-protector-strong -Wformat -Werror=format-security  -flto "
os.environ["PY_CFLAGS"] = "-Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -fstack-protector-strong -Wformat -Werror=format-security  -flto "
os.environ["PY_CORE_CFLAGS"] = "-Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fstack-protector-strong -Wformat -Werror=format-security -flto -fuse-linker-plugin -ffat-lto-objects -std=c99 -Wextra -Wno-unused-result -Wno-unused-parameter -Wno-missing-field-initializers -Wno-cast-function-type -IObjects -IInclude -IPython -I. -I../Include -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -DPy_BUILD_CORE"

if ( os.environ["CC"] != "clang++-8" ):
  os.environ["CFLAGS"] += "-fuse-linker-plugin -ffat-lto-objects"

from distutils.core import setup, Extension
import op_gen
op_gen.main()

# DNNC variables
DNNC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC       = os.path.join(DNNC_ROOT, 'src')
INCLUDES  = os.path.join(DNNC_ROOT, 'include')
ISYSTEM   = os.path.join(DNNC_ROOT, 'packages', 'eigen-eigen-323c052e1731')
OTHER_OBJECTS = [
                 os.path.join(SRC, 'core',      'obj', 'datatypes.o'),
                 os.path.join(SRC, 'operators', 'obj', 'opTypes.o'),
                 os.path.join(SRC, 'graph',     'obj', 'node.o'),
                 os.path.join(SRC, 'graph',     'obj', 'graph.o'),
                 os.path.join(SRC, 'codegen',   'obj', 'cppCodeGen.o')
                ]


NAME    = 'deepC'
EXT     = '_'+NAME
VERSION = 0.1

_DEBUG=False
_DEBUG_LEVEL = 0


long_description = ""
with open("../README.md", "r") as fh:
    long_description = fh.read()

install_requires = []
install_requires.extend([
    'numpy',
    'onnx',
])

dnnc_module = Extension(EXT,
                        language='C++17',
                        sources=['dnnc.i', 'dnnc_api.cpp', 'dnnc_pyutils.cpp'],
                        define_macros=[('NDEBUG', '1')],
                        include_dirs =['../include'],
                        extra_compile_args=['-isystem' + ISYSTEM,
                            '-O3', '-Wall', '-std=c++17', '-fPIC', '-march=native', '-msse2' ],
                        extra_objects=OTHER_OBJECTS,
                        swig_opts=['-c++', '-Wall', '-I'+INCLUDES],
                       )

setup(
    name=NAME,
    version=VERSION,
    description="deepC: Deep Neural Network Compiler",
    long_description=long_description,
    author='Rohit Sharma et. al.',
    author_email='rohit@find-domain-name-on-github.com',
    url='https://github.com/ai-techsystems/dnnCompiler',
    keywords='AITS deepC machine learning',
    classifiers=[
        'Development Status :: 0.1 - prealpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
    ],
    ext_packages = [NAME],
    ext_modules = [dnnc_module],
    py_modules = ['dnnc'],
)
