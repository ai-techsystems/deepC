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

void dnnc::graph::resetNodeMarks() {
  for (node *n : _nodes)
    n->resetMarks();
}

bool dnnc::graph::sanityCheck() {
  bool result = true;
  for (node *n : _nodes) {
    std::vector<node *> next_level_nodes;
    if (false == n->inputNodes(*this, next_level_nodes)) {
      if (n->ntype() != node::INPUT && n->symbol() != opConstant) {
        std::cerr << "ERROR (GRAPH): some of graph " + _name + "'s node " +
                         n->name() + "'s\n";
        std::cerr << "               outputs are not connected to other nodes "
                     "in the graph.\n";
        result = false;
      }
    }
    if (false == n->outputNodes(*this, next_level_nodes)) {
      if (n->ntype() != node::OUTPUT) {
        std::cerr << "ERROR (GRAPH): some of graph " + _name + "'s node " +
                         n->name() + "'s\n";
        std::cerr << "               inputs are not connected to other nodes "
                     "in the graph.\n";
        result = false;
      }
    }
  }
  return result;
}

#ifdef DNNC_GRAPH_TEST
using namespace dnnc;

int main() {
  dnnc::graph &g = dnnc::Graph();

  for (node &n : g) {
    std::cout << n->name() << "\n";
  }

  g.setName("CNTK");
  return 0;
}

#endif
