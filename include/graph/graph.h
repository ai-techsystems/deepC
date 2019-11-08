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
  size_t _nodeIndex = 0; /*!< index for creating names for nodes without name */
  std::vector<node *> _nodes;
  std::vector<size_t> _inputs;  /*!< indices in _nodes containng input nodes */
  std::vector<size_t> _outputs; /*!< indices in _nodes containng output nodes */
  std::vector<dnnParameters> _initializers;

  /*!< Hierarchical graph mechanism by registry.
   * 1. Parent registers every new born in _subgraphs (see subgraph method).
   * 2. Before dying, child deregisters itself from parent's _subgraphs (see
   * destructor).
   * */
  graph *_parent = 0x0;
  std::vector<graph *> _subgraphs;

  graph(graph *parent = 0x0) : _nodeIndex(0), _parent(parent) {}
  // prohibited methods for singleton instance
  graph(const graph &other) = delete;
  graph &operator=(const graph &other) = delete;

  size_t nextIndex() { return ++_nodeIndex; }
  std::string createName() { return "dnnc___" + std::to_string(nextIndex()); }

  ioNode *addIONode(std::string name, DNNC_DataType type,
                    std::vector<size_t> shape, node::NODE_TYPE ntype) {
    assert(ntype == node::INPUT || ntype == node::OUTPUT);
    node *newNode = 0x0;
    if (findNodeByName(name, newNode)) {
      assert((newNode->ntype() == node::INPUT ||
              newNode->ntype() == node::OUTPUT) &&
             "found operator node with same name as io node");
      assert(newNode->symbol() == opInvalid &&
             "found operator node with same name as io node.");
      return dynamic_cast<ioNode *>(newNode);
    }
    name = name.empty() ? createName() : name;
    ioNode *new_ioNode = new ioNode(name, ntype, type, shape);
    _nodes.push_back(new_ioNode);
    ntype == node::INPUT ? _inputs.push_back(_nodes.size() - 1)
                         : _outputs.push_back(_nodes.size() - 1);
    return new_ioNode;
  }

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
  void destroy() {
    if (_parent) {
      // Before dying, deregister itself from parent's _subgraphs.
      // Erase-Remove idiom
      _parent->_subgraphs.erase(std::remove(_parent->_subgraphs.begin(),
                                            _parent->_subgraphs.end(), this),
                                _parent->_subgraphs.end());
    }
    for (auto &sg : _subgraphs)
      delete sg;
    for (auto &n : _nodes)
      delete n;
    _name = "";
    _nodeIndex = 0;
    _nodes.clear();
    _inputs.clear();
    _outputs.clear();
    _initializers.clear();
    _subgraphs.clear();
  }
  ~graph() { destroy(); }
  void setName(std::string name) { _name = name; }

  size_t nNodes() { return _nodes.size(); }
  /*<! reset all existing marks on the graph nodes */
  void resetNodeMarks();

  /*<! add compute node to the graph */
  opNode *addOPNode(std::string name, OPCODE symbol) {
    assert(symbol != opInvalid &&
           "operator node can not be created with invalid opCode.");

    node *newNode = 0x0;
    if (false == name.empty() && findNodeByName(name, newNode)) {
      assert(newNode->ntype() == node::OPERATOR &&
             "found io node with same name as operator node");
      assert(newNode->symbol() != symbol &&
             "found operator node with same name and difference symbol");
      return dynamic_cast<opNode *>(newNode);
    }
    name = name.empty() ? createName() : name;
    opNode *new_opNode = new opNode(symbol, name);
    _nodes.push_back(new_opNode);
    return new_opNode;
  }
  /*<! add input term node to the comptue graph */
  ioNode *addInput(std::string name, DNNC_DataType type,
                   std::vector<size_t> shape) {
    return addIONode(name, type, shape, node::INPUT);
  }
  /*<! add output term node to the comptue graph */
  ioNode *addOutput(std::string name, DNNC_DataType type,
                    std::vector<size_t> shape) {
    return addIONode(name, type, shape, node::OUTPUT);
  }
  std::vector<ioNode *> inputs() {
    std::vector<ioNode *> vins;
    for (size_t &i : _inputs)
      vins.push_back(dynamic_cast<ioNode *>(_nodes[i]));
    return vins;
  }
  std::vector<ioNode *> outputs() {
    std::vector<ioNode *> vouts;
    for (size_t &i : _outputs)
      vouts.push_back(dynamic_cast<ioNode *>(_nodes[i]));
    return vouts;
  }
  bool isOutput(std::string name) {
    for (size_t &i : _outputs)
      if (_nodes[i]->name() == name)
        return true;
    return false;
  }

  void addParameters(dnnParameters param) { _initializers.push_back(param); }
  std::vector<dnnParameters> parameters() { return _initializers; }

  /*<! Search all nodes in the graph. Return a vector of nodes with
   * IO (input or output) same as name passed as argument.*/
  std::vector<node *> findNodesWithIO(std::string name, bool in = true) {
    std::vector<node *> nodes;
    for (node *n : _nodes) {
      if (n->ntype() == node::OPERATOR) {
        if (in) {
          std::vector<std::string> n_ins = dynamic_cast<opNode *>(n)->inputs();
          if (std::find(n_ins.begin(), n_ins.end(), name) != n_ins.end())
            nodes.push_back(n);
        } else {
          std::vector<std::string> n_outs =
              dynamic_cast<opNode *>(n)->outputs();
          if (std::find(n_outs.begin(), n_outs.end(), name) != n_outs.end())
            nodes.push_back(n);
        }
      } else if (n->ntype() == node::INPUT && in == false) {
        if (n->name() == name)
          nodes.push_back(n);
      } else if (n->ntype() == node::OUTPUT && in == true) {
        if (n->name() == name)
          nodes.push_back(n);
      }
    }
    return nodes;
  }
  bool findNodeByName(std::string name, node *&n) {
    for (node *other : _nodes) { // TODO: use std::find
      if (other->name() == name) {
        n = other;
        return true;
      }
    }
    return false;
  }

  bool sanityCheck();

#ifndef SWIGPYTHON
  struct node_iter {
    int pos;
    inline void next(const graph *ref) { ++pos; }
    inline void begin(const graph *ref) { pos = 0; }
    inline void end(const graph *ref) { pos = ref->_nodes.size(); }
    inline node *&get(graph *ref) { return ref->_nodes[pos]; }
    inline const node *get(const graph *ref) { return ref->_nodes[pos]; }
    inline bool cmp(const node_iter &s) const { return pos != s.pos; }
  };
  SETUP_ITERATORS(graph, node *, node_iter)
#endif
};
static graph &Graph() { return graph::singleton(); }
} // namespace dnnc
