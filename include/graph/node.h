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

#include "core/iterator.h"
#include "graph/irData.h"
#include "operators/baseOperator.h"
#include <vector>

namespace dnnc {

// Forward declaration
class graph;

class dnnParameters {
protected:
  std::string _name;
  irTypeData _value;

public:
  dnnParameters(std::string n, irTypeData &v) : _name(n), _value(v) {}
  std::string name() { return _name; }
  irTypeData data() { return _value; }
};

class nodeAttribute {
protected:
  OPATTR _name = attr_invalid;
  irTypeData _value;

public:
  nodeAttribute(OPATTR n, irTypeData &v) : _name(n), _value(v) {}
  OPATTR name() { return _name; }
  irTypeData data() { return _value; }
};

/*! Graph node
 * */
class node {
protected:
  std::string _name;
  // TODO: add node attributes like level, placeholder,
  //       const, variable etc.
public:
  enum NODE_TYPE { NONE = 0, INPUT, OUTPUT, OPERATOR };

  node(std::string n = "") : _name(n) {}
  void setName(std::string n) { _name = n; }
  std::string name() { return _name; }

  virtual OPCODE symbol() { return opInvalid; }
  virtual NODE_TYPE ntype() { return NONE; }
  virtual DNNC_DataType dtype() { return NOTYPE; }
  virtual ~node() {}
};

/*! Compute Graph IO Node.
 *       It represents place holder unit (for inputs and outputs)
 * represented as memory buffer in underlying hardware.
 * */
class ioNode : public node {
protected:
  NODE_TYPE _ntype;
  DNNC_DataType _dtype;
  std::vector<size_t> _shape;

  ioNode() = delete;

public:
  ioNode(std::string n, NODE_TYPE nt, DNNC_DataType dt, std::vector<size_t> shp)
      : node(n), _ntype(nt), _dtype(dt), _shape(shp) {}
  DNNC_DataType dtype() override { return _dtype; }
  NODE_TYPE ntype() override { return _ntype; }
  std::vector<size_t> shape() { return _shape; }
};

/*! Compute Graph operator Node.
 *       It represents basic computational unit (like adder/multiplier)
 * available in underlying hardware.
 * */
class opNode : public node {
protected:
  OPCODE _symbol; /*!< operator aka symbol */
  std::vector<std::string>
      _inputs; /*!< inputs, i.e. tensors coming to   this node */
  // This is a vector of one element, kept for future requirement.
  std::vector<std::string>
      _outputs; /*!< outputs, i.e tensor going  from this node */
  std::vector<nodeAttribute> _attributes; /*!< attributes of the node, i.e.
                                        values that don't flow in and out */

  bool getNodes(graph &, std::vector<node *> &, std::vector<std::string>);
  opNode() = delete; /*!< default constructor not allowed */
public:
  opNode(OPCODE sym, std::string n = "") : node(n), _symbol(sym) {}
  ~opNode() {}

  void addInput(std::string n) { _inputs.push_back(n); }
  void addOutput(std::string n) { _outputs.push_back(n); }
  void addAttribute(nodeAttribute &attr) { _attributes.push_back(attr); }

  OPCODE symbol() override { return _symbol; }
  NODE_TYPE ntype() override { return OPERATOR; }
  DNNC_DataType
  dtype() override { /*!< graph DFS will inference dtype in future. */
    return FLOAT;
  }

  std::vector<std::string> inputs() { return _inputs; }
  std::vector<std::string> outputs() { return _outputs; }
  bool inputNodes(graph &g, std::vector<node *> &nodes) {
    return getNodes(g, nodes, _inputs);
  };
  bool outputNodes(graph &g, std::vector<node *> &nodes) {
    return getNodes(g, nodes, _outputs);
  }

#ifndef SWIGPYTHON
  struct attr_iter {
    int pos;
    inline void next(const opNode *ref) { ++pos; }
    inline void begin(const opNode *ref) { pos = 0; }
    inline void end(const opNode *ref) { pos = ref->_attributes.size(); }
    inline nodeAttribute &get(opNode *ref) { return ref->_attributes[pos]; }
    inline const nodeAttribute &get(const opNode *ref) {
      return ref->_attributes[pos];
    }
    inline bool cmp(const attr_iter &s) const { return pos != s.pos; }
  };
  SETUP_ITERATORS(opNode, nodeAttribute &, attr_iter)
#endif
};

} // namespace dnnc
