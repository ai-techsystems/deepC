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
    if (n->ntype() == node::INPUT) {
      std::vector<node *> opNodes;
      if (false == dynamic_cast<ioNode *>(n)->outputNodes(*this, opNodes)) {
        std::cerr << "ERROR (GRAPH): graph input node(" + n->name() +
                         " is not connected to other nodes in the graph.\n";
        result = false;
      }
    } else if (n->ntype() == node::OUTPUT) {
      std::vector<node *> opNodes;
      if (false == dynamic_cast<ioNode *>(n)->inputNodes(*this, opNodes)) {
        std::cerr << "ERROR (GRAPH): graph output node(" + n->name() +
                         " is not connected to other nodes in the graph.\n";
        result = false;
      }
    } else if (n->ntype() == node::OPERATOR) {
      opNode *oNode = dynamic_cast<opNode *>(n);
      for (auto in : oNode->inputs()) {
        node *newNode = 0x0;
        if (false == findNodeByName(in, newNode)) {
          std::cerr << "ERROR (GRAPH): graph operator node(" + n->name() +
                           ")'s input " + in + " is not found in the graph.\n";
          result = false;
        }
      }
      for (auto out : oNode->outputs()) {
        node *newNode = 0x0;
        if (false == findNodeByName(out, newNode)) {
          std::cerr << "ERROR (GRAPH): graph operator node(" + n->name() +
                           ")'s output " + out +
                           " is not found in the graph.\n";
          result = false;
        }
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
