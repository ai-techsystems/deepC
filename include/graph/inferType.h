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
#include <set>

namespace dnnc {

class inferDataType {
protected:
  graph &_graph;

  bool propagate(opNode &computeNode, DNNC_DataType inType) {

    if (computeNode.isMarked(node::VISITED))
      return true;

    // cycle detected.
    if (computeNode.isMarked(node::VISITING))
      return true;

    computeNode.mark(node::VISITING);

    assert(computeNode.ntype() == node::OPERATOR);
    assert(computeNode.symbol() != opInvalid);

    std::vector<node *> outs;
    if (false == computeNode.outputNodes(_graph, outs)) {
      std::cerr << "ERROR (DIFR): cound not find all nodes for "
                << computeNode.name() << ",\n";
    }

    // infer data type and set it on the node.
    // TODO: do not propagate forward, if old and new dtype are same.
    computeNode.dtype(typePrecedence(inType, computeNode.dtype())
                          ? inType
                          : computeNode.dtype());

    for (auto next : outs) {
      if (next->ntype() == node::OPERATOR)
        propagate(*dynamic_cast<opNode *>(next), computeNode.dtype());
    }

    computeNode.mark(node::VISITED);
    return true;
  }

public:
  inferDataType(graph &graph) : _graph(graph) {}
  bool main() {
    bool inferred = bool(_graph.nNodes());

    _graph.resetNodeMarks();

    for (ioNode *n : _graph.inputs()) {
      std::vector<node *> nextLevelNodes;
      if (n->outputNode(_graph, nextLevelNodes))
        for (node *next : nextLevelNodes) {
          if (next->ntype() == node::OPERATOR) {
            inferred &= propagate(*dynamic_cast<opNode *>(next), n->dtype());
          }
        }
    }

    return inferred;
  }
}; // class inferDataType
} // namespace dnnc
