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
#include <graph/node.h>

namespace dnnc {
/*!< This is a directed graph representing data flow graph
 * for deep neural network. Singleton by design, and light by
 * construction, it lives throughout the life of program and dies
 * with it.
 */
class graph {
protected:
  string _name;
  set<const node, nodeCmp> _nodeSet;
  set<const edge, edgeCmp> _edgeSet;

  // prohibited methods for singleton instance
  graph() {}
  graph(const graph &other) {}
  graph &operator=(const graph &other) {}

public:
  static graph &theGraph() {
    static graph
        _graph; /*!< only one graph can be created, (singleton by design) */
    return _graph;
  }
  void setName(std::string name) { _name = name; }
  bool registerNode(node);
};
graph &theGraph() { return theGraph(); }
} // namespace dnnc
