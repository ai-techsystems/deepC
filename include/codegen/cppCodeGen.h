// Copyright 2018 The AITS DNNC Authors.All Rights Reserved.
//
// Licensed to the Apache Software Foundation(ASF) under one
// or more contributor license agreements.See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.See the License for the
// specific language governing permissionsand limitations
// under the License.
//
// This file is part of AITS DNN compiler maintained at
// https://github.com/ai-techsystems/dnnCompiler
//
#pragma once

#include <graph/graph.h>

namespace dnnc {
class cppCodeGen {
protected:
  std::string _tab = "  ";
  graph &_graph;
  std::string _outFile;
  std::vector<std::string> _includes;

  std::string initializeData(irTypeData, std::string);
  std::string writeIncludes();
  std::string writeMainFunction(std::string);

  std::string nodeName(node *n);

  std::string write(opNode &);
  std::string write(ioNode &);
  std::string writeConstantOperator(opNode &computeNode,
                                    std::vector<node *> &outs);
  std::string writeUnaryOperator(opNode &computeNode, std::vector<node *> &ins,
                                 std::vector<node *> &outs);
  std::string writeBinaryOperator(opNode &computeNode, std::vector<node *> &ins,
                                  std::vector<node *> &outs);
  std::string writeTernaryOperator(opNode &computeNode,
                                   std::vector<node *> &ins,
                                   std::vector<node *> &outs);
  std::string writeCustomOperator(opNode &computeNode, std::vector<node *> &ins,
                                  std::vector<node *> &outs);
  std::string write(dnnParameters);
  std::string write(nodeAttribute &, std::string);

public:
  cppCodeGen(graph &graph, std::string outFile)
      : _graph(graph), _outFile(outFile) {}
  bool write();
};
} // namespace dnnc
