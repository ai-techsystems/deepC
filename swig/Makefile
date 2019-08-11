# Copyright 2018 The DNNC Authors. All Rights Reserved.
# 
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
#
# This file is part of DNN compiler maintained at 
# https://github.com/ai-techsystems/dnnCompiler
#

MAKEFLAGS += -j1

OBJ_DIR=./obj
LIB_DIR=./lib
LIB=$(LIB_DIR)/libdnnc.so

PYTHON=/usr/bin/python3
PYMOD=_dnnc.so

MKDIR_P=mkdir -p
LN_S=ln -s
SWIG=/usr/bin/swig
SWIG_FLAGS=-python -c++ -Wall
DNNC_INCLUDES= -I../include -I../packages/eigen-eigen-323c052e1731
PY_INCLUDES= -I/usr/include/python3.6
GCC=/usr/bin/g++
CPP_FLAGS=-O2 -Wall -std=c++14 -fPIC 
LD=/usr/bin/ld
LD_FLAGS=-shared 

ifneq (,$(findstring $(DEBUG),yY))
CPP_FLAGS=-O2 -Wall -std=c++14 -fPIC -D_DEBUG -g
endif

ifneq (,$(findstring $(LMCHECK),yY))
CPP_FLAGS=-O2 -Wall -std=c++14 -fPIC -D_DEBUG -g -lmcheck
endif

ifneq (,$(findstring $(PROF),yY))
CPP_FLAGS=-pg -Wall -std=c++14 -fPIC 
endif


all:$(LIB) VERIFY

SWIG_MODULES=dnnc.i
SWIG_CPPS=$(SWIG_MODULES:%.i=%_swig.cpp)
$(SWIG_CPPS):$(SWIG_MODULES)
	$(SWIG) $(SWIG_FLAGS) $(DNNC_INCLUDES) -o $@ $<

CPP_SRCS=$(SWIG_CPPS) dnnc_api.cpp
OBJECTS=$(CPP_SRCS:%.cpp=$(OBJ_DIR)/%.o)
$(OBJ_DIR)/%.o:%.cpp
	test -d $(OBJ_DIR) || $(MKDIR_P) $(OBJ_DIR) 
	@echo "compiling $<"
	$(GCC) $(CPP_FLAGS) $(PY_INCLUDES) $(DNNC_INCLUDES) -c $< -o $@

$(LIB):$(OBJECTS)
	test -d $(LIB_DIR) || $(MKDIR_P) $(LIB_DIR) 
	$(GCC) $(LD_FLAGS) $(OBJECTS) -o $@
	$(LN_S) -f $@ $(PYMOD)

VERIFY:
	$(PYTHON) ../test/swig/basic.py

clean:
	\rm -fr $(OBJ_DIR) $(LIB_DIR) dnnc_swig.cpp dnnc.py dnnc.pyc $(PYMOD) __pycache__

.DEFAULT: all
.PHONY: print_vars

print_vars:
	@echo $(SWIG_CPPS)
	@echo $(CPP_SRCS)
	@echo $(OBJECTS)
