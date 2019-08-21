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

MKDIR_P=mkdir -p
OBJ_DIR=./obj
LIB_DIR=./lib
LIB=$(LIB_DIR)/liboperators.so

INCLUDES= -I../../include -isystem ../../packages/eigen-eigen-323c052e1731
GCC=/usr/bin/g++
CPP_FLAGS=-O2 -Wall -std=c++14 -fPIC -D_DEBUG
LD=/usr/bin/ld
LD_FLAGS=-shared 

all:$(LIB)

CPP_SRCS=$(wildcard *.cpp)
OBJECTS=$(CPP_SRCS:%.cpp=$(OBJ_DIR)/%.o)
$(OBJ_DIR)/%.o:%.cpp
	test -d $(OBJ_DIR) || $(MKDIR_P) $(OBJ_DIR) 
	@echo "compiling $<"
	$(GCC) $(CPP_FLAGS) $(INCLUDES) -c $< -o $@

$(LIB):$(OBJECTS)
	test -d $(LIB_DIR) || $(MKDIR_P) $(LIB_DIR) 
	$(GCC) $(LD_FLAGS) $(OBJECTS) -o $@

clean:
	\rm -fr $(OBJ_DIR) $(LIB_DIR) 

.DEFAULT: all
.PHONY: print_vars

print_vars:
	@echo $(SWIG_CPPS)
	@echo $(CPP_SRCS)
	@echo $(OBJECTS)
