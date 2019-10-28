# Copyright 2018 The DNNC Authors. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
#
# This file is part of DNN compiler maintained at
# https://github.com/ai-techsystems/dnnCompiler
#

import os, sys

import deepC.dnnc as dnnc

import onnx
import struct


class pbReader :
  """Reader class for DNNC models in ONNX binary/protobuf format."""

  def __init__(self):
      dnncModule = sys.modules.get('deepC.dnnc')
      if ( dnncModule is None ) :
        print("ERROR (DNNC): could not find dnnc module. Please make sure dnnc is imported before calling ", __name__)
      self._dcGraph = None ;
      self._bundleDir = None;
      self._writeParamToDisk = True ;

  def __del__(self):
      del self._dcGraph ;

  def writeParamsToFile(self, name, data):

    str_data = '\n'.join([str(d) for d in data])
    if ( len(str_data) == 0 ):
      print("ERROR (ONNX): did not find data for initializer ", name);
      return

    paramFile = os.path.join(self._bundleDir,name)
    print("INFO (ONNX): writing model parameter " + name + " to dir " + self._bundleDir + ".");
    with open(paramFile, "w") as fp:
      fp.write(str_data)
      fp.close();

  def addParams(self, param):

      if ( param is None ):
        return None;

      if ( len(param.FindInitializationErrors()) > 0 ):
        print("WARNING (ONNX): initializer " + param.name + " has following errors.\n");
        print("               ", param.FindInitializationErrors());
        print("                trying to load data with errors.\n");

      param_type = dnnc.IR_DataType_NOTYPE;
      param_shape = dnnc.vectorSizeT(param.dims)
      param_vec  = None
      param_vals = None
      if param.data_type == param.INT8 :
        param_type = dnnc.IR_DataType_INT8;
        param_vals = [int(n) for n in param.int32_data]
        if ( len(param_vals) == 0 ):
            param_vals = [int(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorInt()
        else:
          param_vec = dnnc.vectorInt(param_vals)
      elif param.data_type == param.INT16 :
        param_type = dnnc.IR_DataType_INT16;
        param_vals = [int(n) for n in param.int32_data]
        if ( len(param_vals) == 0 ):
            param_vals = [int(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorInt()
        else:
          param_vec = dnnc.vectorInt(param_vals)
      elif param.data_type == param.INT32:
        param_type = dnnc.IR_DataType_INT32;
        param_vals = [int(n) for n in param.int32_data]
        if ( len(param_vals) == 0 ):
            param_vals = [int(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorInt()
        else:
          param_vec = dnnc.vectorInt(param_vals)
      elif param.data_type == param.INT64:
        param_type = dnnc.IR_DataType_INT64;
        param_vals = [int(n) for n in param.int64_data]
        if ( len(param_vals) == 0 ):
            param_vals = [int(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorInt()
        else:
          param_vec = dnnc.vectorInt(param_vals)
      elif param.data_type == param.UINT8 :
        param_type = dnnc.IR_DataType_UINT8;
        param_vals = [int(n) for n in param.uint64_data]
        if ( len(param_vals) == 0 ):
            param_vals = [int(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorInt()
        else:
          param_vec = dnnc.vectorInt(param_vals)
      elif param.data_type == param.UINT16 :
        param_type = dnnc.IR_DataType_UINT16;
        param_vals = [int(n) for n in param.uint64_data]
        if ( len(param_vals) == 0 ):
            param_vals = [int(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorInt()
        else:
          param_vec = dnnc.vectorInt(param_vals)
      elif param.data_type == param.UINT32:
        param_type = dnnc.IR_DataType_UINT32;
        param_vals = [int(n) for n in param.uint64_data]
        if ( len(param_vals) == 0 ):
            param_vals = [int(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorInt()
        else:
          param_vec = dnnc.vectorInt(param_vals)
      elif param.data_type == param.UINT64:
        param_type = dnnc.IR_DataType_UINT64;
        param_vals = [int(n) for n in param.uint64_data]
        if ( len(param_vals) == 0 ):
            param_vals = [int(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorInt()
        else:
          param_vec = dnnc.vectorInt(param_vals)
      elif param.data_type == param.FLOAT16 :
        param_type = dnnc.IR_DataType_FLOAT16;
        param_vals = [float(n) for n in param.float_data]
        if ( len(param_vals) == 0 ):
            param_vals = [float(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorInt()
        else:
          param_vec = dnnc.vectorFloat(param_vals)
      elif param.data_type == param.BFLOAT16 :
        param_type = dnnc.IR_DataType_BFLOAT16;
        param_vals = [float(n) for n in param.float_data]
        if ( len(param_vals) == 0 ):
            param_vals = [float(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorFloat()
        else:
          param_vec = dnnc.vectorFloat(param_vals)
      elif param.data_type == param.FLOAT:
        param_type = dnnc.IR_DataType_FLOAT;
        param_vals = [float(n) for n in param.float_data]
        if ( len(param_vals) == 0 ):
            param_vals = [float(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorFloat()
        else:
          param_vec = dnnc.vectorFloat(param_vals)
      elif param.data_type == param.DOUBLE:
        param_type = dnnc.IR_DataType_DOUBLE;
        param_vals = [float(n) for n in param.double_data]
        if ( len(param_vals) == 0 ):
            param_vals = [float(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorFloat()
        else:
          param_vec = dnnc.vectorFloat(param_vals)
      elif param.data_type == param.STRING:
        param_type = dnnc.IR_DataType_STRING;
        param_vals = [str(s) for s in param.string_data]
        if ( len(param_vals) == 0 ):
            param_vals = [str(n) for n in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorStr()
        else:
          param_vec = dnnc.vectorStr(param_vals)
      elif param.data_type == param.BOOL:
        param_type = dnnc.IR_DataType_BOOL;
        param_vals = [bool(b) for b in param.raw_data]
        if ( len(param_vals) == 0 ):
            param_vals = [bool(b) for b in param.raw_data]
        if ( self._writeParamToDisk ) :
          self.writeParamsToFile(param.name, param_vals);
          param_vec = dnnc.vectorBool()
        else:
          param_vec = dnnc.vectorBool(param_vals)
      else:
        print("ERROR (ONNX): graph-node " + node.name + "\'s attribute " + \
               param.name + " type " + str(param.data_type) + " is not valid.")

      if ( self._writeParamToDisk is False and
              (param_type is dnnc.IR_DataType_NOTYPE or param_vec is None or param_vec.size()==0) ) :
        print("ERROR (ONNX): did not find data for initializer ", param.name);
        return;

      param_irData = dnnc.irTypeData(param_type, param_vec) ;
      dnnc_param  = dnnc.dnnParameters(param.name, param_shape, param_irData);
      self._dcGraph.addParameters(dnnc_param) ;

      return dnnc_param;


  def addOPNode(self, node):

    op_type = dnnc.getOpCode(node.op_type);
    if ( op_type is dnnc.opInvalid ):
      print("ERROR (ONNX):" +  node.op_type +" is not a valid graph-node op type.")
      return None

    dcNode = self._dcGraph.addOPNode(node.name, op_type);

    for nd in node.input:
      dcNode.addInput(nd)

    for nd in node.output:
      dcNode.addOutput(nd)

    for attr in node.attribute:
      attr_type = dnnc.IR_DataType_NOTYPE;
      attr_vals = []
      attr_vec  = None
      if attr.type == onnx.AttributeProto.INT:
        attr_type = dnnc.IR_DataType_INT32;
        attr_vals.append(attr.i)
        attr_vec = dnnc.vectorInt(attr_vals)
      elif attr.type == onnx.AttributeProto.INTS:
        attr_type = dnnc.IR_DataType_INT32;
        for val in attr.ints:
          attr_vals.append(int(val))
        attr_vec = dnnc.vectorInt(attr_vals)
      elif attr.type == onnx.AttributeProto.FLOAT:
        attr_type = dnnc.IR_DataType_FLOAT;
        attr_vals.append(attr.f)
        attr_vec = dnnc.vectorFloat(attr_vals)
      elif attr.type == onnx.AttributeProto.FLOATS:
        attr_type = dnnc.IR_DataType_FLOAT;
        for val in attr.floats:
          attr_vals.append(float(val))
        attr_vec = dnnc.vectorFloat(attr_vals)
      elif attr.type == onnx.AttributeProto.STRING:
        attr_type = dnnc.IR_DataType_STRING;
        attr_vals.append(str(attr.s))
        attr_vec = dnnc.vectorStr(attr_vals)
      elif attr.type == onnx.AttributeProto.STRINGS:
        attr_type = dnnc.IR_DataType_STRING;
        for val in attr.strings:
          attr_vals.append(str(val))
        attr_vec = dnnc.vectorStr(attr_vals)
      elif attr.type == onnx.AttributeProto.TENSOR:
        if ( attr.t.data_type == onnx.TensorProto.INT8  or
             attr.t.data_type == onnx.TensorProto.INT16 or
             attr.t.data_type == onnx.TensorProto.INT32 or
             attr.t.data_type == onnx.TensorProto.INT64   ) :

          attr_type = attr.t.data_type
          attr_data = None;
          pack_format = 'P';
          if ( attr.t.data_type == onnx.TensorProto.INT8 ) :
            pack_format = 'b'
          if ( attr.t.data_type == onnx.TensorProto.INT16) :
            pack_format = 'h'
          if ( attr.t.data_type == onnx.TensorProto.INT32) :
            if ( attr.t.int32_data ):
              attr_data = attr.t.int32_data
            pack_format = 'i'
          if ( attr.t.data_type == onnx.TensorProto.INT64) :
            if ( attr.t.int64_data ):
              attr_data = attr.t.int64_data
            pack_format = 'q'

          if ( attr_data is None ) :
            len=1
            for d in attr.t.dims:
              len *= d
            attr_data = struct.unpack(pack_format*len, attr.t.raw_data) ;

          if ( attr_data is not None ) :
            attr_tensor = dnnc.intTensor(attr.t.dims, attr.name)
            attr_tensor.load(attr_data);
            attr_vec = dnnc.vectorTensorInt()
            attr_vec.push_back(attr_tensor)
          else:
            print("ERROR (ONNX): could not extract data for graph-node " + \
                    node.name + "\'s attribute " +  attr.name + ".\n");

        elif ( attr.t.data_type == onnx.TensorProto.FLOAT16 or
             attr.t.data_type == onnx.TensorProto.FLOAT   or
             attr.t.data_type == onnx.TensorProto.DOUBLE    ):

          attr_type = attr.t.data_type
          attr_data = None;
          pack_format = 'P';
          if ( attr.t.data_type == onnx.TensorProto.FLOAT16 ) :
            if ( attr.t.float_data ):
              attr_data = attr.t.float_data
            pack_format = 'e'
          if ( attr.t.data_type == onnx.TensorProto.FLOAT ) :
            if ( attr.t.float_data ):
              attr_data = attr.t.float_data
            pack_format = 'f'
          if ( attr.t.data_type == onnx.TensorProto.DOUBLE ) :
            if ( attr.t.double_data ):
              attr_data = attr.t.double_data
            pack_format = 'd'

          if ( attr_data is None ) :
            len=1
            for d in attr.t.dims:
              len *= d
            attr_data = struct.unpack(pack_format*len, attr.t.raw_data) ;

          if ( attr_data is not None ):
            attr_tensor = dnnc.floatTensor(attr.t.dims, attr.name)
            attr_tensor.load(attr_data);
            attr_vec = dnnc.vectorTensorFloat()
            attr_vec.push_back(attr_tensor)
          else:
            print("ERROR (ONNX): could not extract data for graph-node " + \
                    node.name + "\'s attribute " +  attr.name + ".\n");
        else:
          print("ERROR (ONNX): attribute tensor's datatype " + str(attr.t.data_type) +
                  " isn't understood.")

      elif attr.type == onnx.AttributeProto.TENSORS:
        attr_type = dnnc.IR_DataType_TENSORS;
        attr_vals.append(attr.tensors)
        attr_vec = dnnc.vectorTensorFloat(dnnc.floatTensor(attr_vals))
      elif attr.type == onnx.AttributeProto.GRAPH:
        attr_type = dnnc.IR_DataType_GRAPH;
        attr_vals.append(attr.g)
        print("ERROR (ONNX): sub-graph in graph-node is not yet supported.")
      elif attr.type == onnx.AttributeProto.GRAPHS:
        attr_type = dnnc.IR_DataType_GRAPH;
        attr_vals.append(attr.graphs)
        print("ERROR (ONNX): sub-graph in graph-node is not yet supported.")
      else:
        print("ERROR (ONNX): graph-node " + node.name + "\'s attribute " + \
               attr.name + " type " + str(attr.type) + " is not valid.")
        continue

      if ( attr_type is dnnc.IR_DataType_NOTYPE or attr_vec is None or attr_vec.size() == 0 ) :
        print("ERROR (ONNX): graph-node " + node.name + "\'s attribute " + \
               attr.name + " has no data.")
        continue ;

      attr_code = dnnc.getAttrName(attr.name);
      if ( attr_code is dnnc.attr_invalid ):
        print("WARN (ONNX): " + attr.name + " is not a valid graph-node attribute.")
        print("             operator " + node.op_type + " will be added without this attribute." )

      cAttrData = dnnc.irTypeData(attr_type,attr_vec) ;
      cAttr = dnnc.nodeAttribute(attr_code, cAttrData);
      dcNode.addAttribute(cAttr);


    return dcNode;

  def createTermNode(self, term):
    term_name  = term.name
    data_type  = dnnc.NOTYPE
    term_shape = []
    if ( term.type.tensor_type.elem_type ) :
      data_type  = term.type.tensor_type.elem_type
      if ( data_type <= dnnc.NOTYPE and data_type >= dnnc.TENSOR ) :
        print("ERROR (ONNX):  Term " + term_name + "\'s type " + data_type + " is not valid"  ) ;
        return None ;

    if ( term.type.tensor_type and term.type.tensor_type.shape ) :
      shape = term.type.tensor_type.shape.dim
      for dim in shape:
        if ( dim.dim_param ):
          if ( dim.dim_param == 'None' ):
              term_shape.append(0);
          else:
              print("ERROR (ONNX): terminal (input/output) " + term_name + "\'s dim_param "
                      + dim.dim_param + " is not recognized.");
        elif ( dim.dim_value ) :
          term_shape.append(dim.dim_value)
        else:
          print("ERROR (ONNX): terminal (input/output) " + term_name + " has no dim_param or dim_value")

    return (term_name, data_type, term_shape)

  def main(self, onnx_filename, bundle_dir=None, checker=False, optimize=False):
    dnncModule = sys.modules.get('deepC.dnnc')
    if ( dnncModule is None ) :
      print("ERROR (DNNC): could not find dnnc module. Please make sure dnnc is imported before calling ", __name__)
      return ;

    print("reading onnx model from file ", onnx_filename)

    self._bundleDir = bundle_dir
    if ( self._bundleDir is None ) :
        self._bundleDir = os.path.dirname(onnx_filename);

    model = onnx.load(onnx_filename)
    print("Model info:\n  ir_vesion : ", model.ir_version, "\n  doc       :", model.doc_string)

    if ( optimize ) :
        print("  Optimization enabled.")
        from onnx import optimizer

        for opt_pass in optimizer.get_available_passes():
            print('    running optimization step : {}'.format(opt_pass.replace("_", " ")))
            try :
                model = optimizer.optimize(model, [opt_pass]);
            except Exception as e:
                print ("        optimization failed." + str(e) + "\n. Abandoning and trying next.");
        print ("  optimization done.")

    if ( checker ) :
        try:
            print ("running ONNX model shape inference engine and verification");
            onnx.checker.check_model(model)
            from onnx import shape_inference
            model = shape_inference.infer_shapes(model)
            onnx.checker.check_model(model)
        except Exception as e:
            print ("        failed. moving to next step." + str(e));


    graph = model.graph

    self._dcGraph = dnnc.Graph();
    self._dcGraph.setName(graph.name)

    nodes = graph.node
    for node in nodes:
      dcNode = self.addOPNode(node);

    for terminal in graph.input:
      dcTerm = self.createTermNode(terminal);
      if ( dcTerm != None and len(dcTerm) == 3 ):
        self._dcGraph.addInput(dcTerm[0], dcTerm[1], dcTerm[2]);

    for terminal in graph.output:
      dcTerm = self.createTermNode(terminal);
      if ( dcTerm != None and len(dcTerm) == 3 ):
        self._dcGraph.addOutput(dcTerm[0], dcTerm[1], dcTerm[2]);

    for param in graph.initializer:
      self.addParams(param);

    try:
        print("running DNNC graph sanity check.");
        if ( False == self._dcGraph.sanityCheck() ):
            print("        FAILED. Please check your model.");
    except Exception as e:
        print ("        FAILED.\n" + str(e));

    return self._dcGraph


def main():
  onnx_file = None
  if len(sys.argv) >= 2:
    onnx_file = sys.argv[1]

  if ( onnx_file is None ) :
    print("\nUsage: "+sys.argv[0]+ " <onnx_model_file>.onnx [bundle_dir]\n")
    exit(0)

  bundle_dir = None
  if len(sys.argv) >= 3:
    bundle_dir = sys.argv[2]
  else:
    bundle_dir = os.path.dirname(onnx_filename);

  parser = pbReader()
  parser.main(onnx_file, bundle_dir, checker=False, optimize=False)

if __name__ == "__main__":
  main()
