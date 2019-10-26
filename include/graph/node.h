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

#include "core/flag.h"
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
  std::vector<DIMENSION> _shape;
  irTypeData _value;

public:
  dnnParameters(std::string n, std::vector<DIMENSION> shape, irTypeData &v)
      : _name(n), _shape(shape), _value(v) {}
  std::string name() { return _name; }
  std::vector<DIMENSION> shape() { return _shape; }
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
  flag _properties; // used in other algorithms like DFS, TopoSort etc.
public:
  enum NODE_TYPE { NONE = 0, INPUT, OUTPUT, OPERATOR };
  enum NODE_PROP { NOT_VISITED = 0, VISITING, VISITED };

  // properties methods.
  void mark(short prop) { _properties.set(prop); }
  void unmark(short prop) { _properties.reset(prop); }
  bool isMarked(short prop) const { return _properties.get(prop); }
  void resetMarks() { _properties = 0; }

  node(std::string n = "") : _name(n) {}
  void setName(std::string n) { _name = n; }
  std::string name() { return _name; }

  virtual OPCODE symbol() { return opInvalid; }
  virtual NODE_TYPE ntype() { return NONE; }
  virtual DNNC_DataType dtype() { return NOTYPE; }
  virtual bool inputNodes(graph &g, std::vector<node *> &nodes) = 0;
  virtual bool outputNodes(graph &g, std::vector<node *> &nodes) = 0;
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

  bool getNodes(graph &, std::vector<node *> &, bool input = true);

public:
  ioNode(std::string n, NODE_TYPE nt, DNNC_DataType dt, std::vector<size_t> shp)
      : node(n), _ntype(nt), _dtype(dt), _shape(shp) {}
  DNNC_DataType dtype() override { return _dtype; }
  NODE_TYPE ntype() override { return _ntype; }
  std::vector<size_t> shape() { return _shape; }

  bool outputNodes(graph &g, std::vector<node *> &nodes) override {
    return getNodes(g, nodes, true);
  };
  bool inputNodes(graph &g, std::vector<node *> &nodes) override {
    return getNodes(g, nodes, false);
  }
};

/*! Compute Graph operator Node.
 *       It represents basic computational unit (like adder/multiplier)
 * available in underlying hardware.
 * */
class opNode : public node {
protected:
  OPCODE _symbol = opInvalid;    /*!< operator aka symbol */
  DNNC_DataType _dtype = NOTYPE; /*<! inferred data type for outputs */
  std::vector<std::string>
      _inputs; /*!< inputs, i.e. tensors coming to   this node */
  std::vector<std::string>
      _outputs; /*!< outputs, i.e tensor going  from this node */
  std::vector<nodeAttribute> _attributes; /*!< attributes of the node, i.e.
                                        values that don't flow in and out */

  bool getNodes(graph &, std::vector<node *> &, bool input = true);
  opNode() = delete; /*!< default constructor not allowed */
public:
  opNode(OPCODE sym, std::string n = "") : node(n), _symbol(sym) {}
  ~opNode() {}

  void addInput(std::string n) { _inputs.push_back(n); }
  void addOutput(std::string n) { _outputs.push_back(n); }
  void addAttribute(nodeAttribute &attr) {
    _attributes.push_back(attr);
    if (_symbol == opConstant && attr.name() == attr_value) {
      IR_DataType data_type = attr.data().type();
      if (data_type == IR_DataType::TENSOR_BOOL)
        _dtype = BOOL;
      else if (data_type == IR_DataType::TENSOR_INT)
        _dtype = INT64;
      else if (data_type == IR_DataType::TENSOR_FLOAT)
        _dtype = DOUBLE;
      else
        _dtype = static_cast<dnnc::DNNC_DataType>(data_type);
    }
  }

  OPCODE symbol() override { return _symbol; }
  NODE_TYPE ntype() override { return OPERATOR; }

  /*!< inferred dtype. */
  void dtype(DNNC_DataType dtype) { _dtype = dtype; }
  DNNC_DataType dtype() override { return _dtype; }

  std::vector<std::string> inputs() { return _inputs; }
  std::vector<std::string> outputs() { return _outputs; }
  bool inputNodes(graph &g, std::vector<node *> &nodes) override {
    return getNodes(g, nodes, true);
  };
  bool outputNodes(graph &g, std::vector<node *> &nodes) override {
    return getNodes(g, nodes, false);
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
