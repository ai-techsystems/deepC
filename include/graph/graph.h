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
#include <set>

namespace dnnc {
struct placeHolder {
public:
  std::string name;
  DNNC_DataType type;
  std::vector<size_t> shape;
  placeHolder(std::string n, DNNC_DataType ty, std::vector<size_t> shp)
      : name(n), type(ty), shape(shp) {}
};

/*!< This is a directed graph representing data flow graph
 * for deep neural networks. Default graph is singleton by design,
 * and light by construction. Default graph lives throughout the
 * life of program and dies with it.
 *
 * One can create subgraphs pointers owned by default graph.
 *
 * Reference: https://github.com/onnx/onnx/blob/rel-1.5.0/docs/IR.md
 */
class graph {
protected:
  std::string _name = "";
  std::vector<node> _nodes;
  std::vector<placeHolder> _inputs;
  std::vector<placeHolder> _outputs;
  std::vector<nodeAttribute> _initializers;

  /*!< Hierarchical graph mechanism by registry.
   * 1. Parent registers every new born in _subgraphs (see subgraph method).
   * 2. Before dying, child deregisters itself from parent's _subgraphs (see
   * destructor).
   * */
  graph *_parent = 0x0;
  std::vector<graph *> _subgraphs;

  graph(graph *parent = 0x0) : _parent(parent) {}
  // prohibited methods for singleton instance
  graph(const graph &other) = delete;
  graph &operator=(const graph &other) = delete;

public:
  static graph &singleton() {
    static graph
        _graph; /*!< only one graph can be created, (singleton by design) */
    return _graph;
  }
  graph &subgraph() {
    graph *sg = new graph(this);
    // register new born in _subgraphs.
    _subgraphs.push_back(sg);
    return *sg;
  }
  ~graph() {
    if (_parent) {
      // Before dying, deregister itself from parent's _subgraphs.
      // Erase-Remove idiom
      _parent->_subgraphs.erase(std::remove(_parent->_subgraphs.begin(),
                                            _parent->_subgraphs.end(), this),
                                _parent->_subgraphs.end());
    }
    for (auto &sg : _subgraphs)
      delete sg;
  }
  void setName(std::string name) { _name = name; }
  void addNode(node n) { _nodes.push_back(n); }
  void addInput(placeHolder in) { _inputs.push_back(in); }
  void addOutput(placeHolder out) { _outputs.push_back(out); }
  void addInitializer(nodeAttribute param) { _initializers.push_back(param); }
};
static graph &Graph() { return graph::singleton(); }
} // namespace dnnc
