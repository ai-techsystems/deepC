# add PYTHONPATH etc here.

import os,sys
DNNC_ROOT=os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..')

# 1. os.environ is needed to launch compiler commands.
if ( "PYTHONPATH" in os.environ ) :
    os.environ["PYTHONPATH"] += os.pathsep + DNNC_ROOT
else:
    os.environ["PYTHONPATH"] = DNNC_ROOT

# 2. sys.path is needed to import deepC
sys.path.append(DNNC_ROOT);
