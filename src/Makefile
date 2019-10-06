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

# CORE must be first, all others depend on CORE
MAKEFLAGS += -j8

all:CORE OPERATORS GRAPH CODEGEN

FORCE:

CORE: FORCE
	$(MAKE) -C core

OPERATORS: CORE
	$(MAKE) -C operators

GRAPH: OPERATORS
	$(MAKE) -C graph

CODEGEN: GRAPH
	$(MAKE) -C codegen

clean:
	$(MAKE) -C core clean
	$(MAKE) -C operators clean
	$(MAKE) -C graph clean
	$(MAKE) -C codegen clean

.PHONY: print_vars

print_vars:
	@echo 
