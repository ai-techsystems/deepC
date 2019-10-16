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

#include "graph/graph.h"

using namespace dnnc;

bool dnnc::ioNode::getNodes(graph &g, std::vector<node *> &nodes, bool input) {
  nodes = g.findNodesWithIO(_name, input);
  return bool(nodes.size());
}

bool dnnc::opNode::getNodes(graph &g, std::vector<node *> &nodes, bool input) {
  std::vector<std::string> names = input ? _inputs : _outputs;
  bool result = bool(names.size());
  for (std::string name : names) {
    std::vector<node *> newNodes = g.findNodesWithIO(name, !input);
    nodes.insert(nodes.end(), newNodes.begin(), newNodes.end());
    result &= bool(newNodes.size());
  }
  return result;
}

#ifdef DNNC_NODE_TEST
#include "operators/Add.h"
using namespace dnnc;

int main() {
  Add<float, float> *op = new Add<float, float>("graph node");
  baseOperator<float, float, float> *bop = op;
  node add1(op);
  node add2(bop);
  std::cout << bop << std::endl;
  return 0;
}

#endif
