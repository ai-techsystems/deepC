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
import onnx
import struct

def dnncOpCode(sym):
  if (sym=="Abs" ):
    return dc.opAbs;
  if (sym=="Acos" ):
    return dc.opAcos;
  if (sym=="Acosh" ):
    return dc.opAcosh;
  if (sym=="Add" ):
    return dc.opAdd;
  if (sym=="And" ):
    return dc.opAnd;
  if (sym=="ArgMax" ):
    return dc.opArgMax;
  if (sym=="ArgMin" ):
    return dc.opArgMin;
  if (sym=="Asin" ):
    return dc.opAsin;
  if (sym=="Asinh" ):
    return dc.opAsinh;
  if (sym=="Atan" ):
    return dc.opAtan;
  if (sym=="Atanh" ):
    return dc.opAtanh;
  if (sym=="AveragePool" ):
    return dc.opAveragePool;
  if (sym=="BatchNormalization" ):
    return dc.opBatchNormalization;
  if (sym=="BitShift" ):
    return dc.opBitShift;
  if (sym=="Cast" ):
    return dc.opCast;
  if (sym=="Ceil" ):
    return dc.opCeil;
  if (sym=="Clip" ):
    return dc.opClip;
  if (sym=="Compress" ):
    return dc.opCompress;
  if (sym=="Concat" ):
    return dc.opConcat;
  if (sym=="Constant" ):
    return dc.opConstant;
  if (sym=="ConstantOfShape" ):
    return dc.opConstantOfShape;
  if (sym=="Conv" ):
    return dc.opConv;
  if (sym=="ConvInteger" ):
    return dc.opConvInteger;
  if (sym=="ConvTranspose" ):
    return dc.opConvTranspose;
  if (sym=="Cos" ):
    return dc.opCos;
  if (sym=="Cosh" ):
    return dc.opCosh;
  if (sym=="CumSum" ):
    return dc.opCumSum;
  if (sym=="DepthToSpace" ):
    return dc.opDepthToSpace;
  if (sym=="DequantizeLinear" ):
    return dc.opDequantizeLinear;
  if (sym=="Div" ):
    return dc.opDiv;
  if (sym=="Dropout" ):
    return dc.opDropout;
  if (sym=="Elu" ):
    return dc.opElu;
  if (sym=="Equal" ):
    return dc.opEqual;
  if (sym=="Erf" ):
    return dc.opErf;
  if (sym=="Exp" ):
    return dc.opExp;
  if (sym=="Expand" ):
    return dc.opExpand;
  if (sym=="EyeLike" ):
    return dc.opEyeLike;
  if (sym=="Flatten" ):
    return dc.opFlatten;
  if (sym=="Floor" ):
    return dc.opFloor;
  if (sym=="FloorDiv" ):
    return dc.opFloorDiv;
  if (sym=="GRU" ):
    return dc.opGRU;
  if (sym=="Gather" ):
    return dc.opGather;
  if (sym=="Gemm" ):
    return dc.opGemm;
  if (sym=="GlobalAveragePool" ):
    return dc.opGlobalAveragePool;
  if (sym=="GlobalLpPool" ):
    return dc.opGlobalLpPool;
  if (sym=="GlobalMaxPool" ):
    return dc.opGlobalMaxPool;
  if (sym=="Greater" ):
    return dc.opGreater;
  if (sym=="GreaterEqual" ):
    return dc.opGreaterEqual;
  if (sym=="HardSigmoid" ):
    return dc.opHardSigmoid;
  if (sym=="Hardmax" ):
    return dc.opHardmax;
  if (sym=="Identity" ):
    return dc.opIdentity;
  if (sym=="If" ):
    return dc.opIf;
  if (sym=="InstanceNormalization" ):
    return dc.opInstanceNormalization;
  if (sym=="IsInf" ):
    return dc.opIsInf;
  if (sym=="IsNaN" ):
    return dc.opIsNaN;
  if (sym=="LRN" ):
    return dc.opLRN;
  if (sym=="LSTM" ):
    return dc.opLSTM;
  if (sym=="LeakyRelu" ):
    return dc.opLeakyRelu;
  if (sym=="Less" ):
    return dc.opLess;
  if (sym=="LessEqual" ):
    return dc.opLessEqual;
  if (sym=="Log" ):
    return dc.opLog;
  if (sym=="LogSoftmax" ):
    return dc.opLogSoftmax;
  if (sym=="Loop" ):
    return dc.opLoop;
  if (sym=="LpNormalization" ):
    return dc.opLpNormalization;
  if (sym=="LpPool" ):
    return dc.opLpPool;
  if (sym=="MatMul" ):
    return dc.opMatMul;
  if (sym=="MatMulInteger" ):
    return dc.opMatMulInteger;
  if (sym=="Max" ):
    return dc.opMax;
  if (sym=="MaxPool" ):
    return dc.opMaxPool;
  if (sym=="MaxRoiPool" ):
    return dc.opMaxRoiPool;
  if (sym=="MaxUnpool" ):
    return dc.opMaxUnpool;
  if (sym=="Mean" ):
    return dc.opMean;
  if (sym=="Min" ):
    return dc.opMin;
  if (sym=="Mod" ):
    return dc.opMod;
  if (sym=="Mul" ):
    return dc.opMul;
  if (sym=="Multinomial" ):
    return dc.opMultinomial;
  if (sym=="Neg" ):
    return dc.opNeg;
  if (sym=="NonMaxSuppression" ):
    return dc.opNonMaxSuppression;
  if (sym=="NonZero" ):
    return dc.opNonZero;
  if (sym=="Not" ):
    return dc.opNot;
  if (sym=="NotEqual" ):
    return dc.opNotEqual;
  if (sym=="OneHot" ):
    return dc.opOneHot;
  if (sym=="Or" ):
    return dc.opOr;
  if (sym=="PRelu" ):
    return dc.opPRelu;
  if (sym=="Pad" ):
    return dc.opPad;
  if (sym=="Pow" ):
    return dc.opPow;
  if (sym=="QLinearConv" ):
    return dc.opQLinearConv;
  if (sym=="QLinearMatMul" ):
    return dc.opQLinearMatMul;
  if (sym=="QuantizeLinear" ):
    return dc.opQuantizeLinear;
  if (sym=="RNN" ):
    return dc.opRNN;
  if (sym=="RandomNormal" ):
    return dc.opRandomNormal;
  if (sym=="RandomNormalLike" ):
    return dc.opRandomNormalLike;
  if (sym=="RandomUniform" ):
    return dc.opRandomUniform;
  if (sym=="RandomUniformLike" ):
    return dc.opRandomUniformLike;
  if (sym=="Reciprocal" ):
    return dc.opReciprocal;
  if (sym=="ReduceL1" ):
    return dc.opReduceL1;
  if (sym=="ReduceL2" ):
    return dc.opReduceL2;
  if (sym=="ReduceLogSum" ):
    return dc.opReduceLogSum;
  if (sym=="ReduceLogSumExp" ):
    return dc.opReduceLogSumExp;
  if (sym=="ReduceMax" ):
    return dc.opReduceMax;
  if (sym=="ReduceMean" ):
    return dc.opReduceMean;
  if (sym=="ReduceMin" ):
    return dc.opReduceMin;
  if (sym=="ReduceProd" ):
    return dc.opReduceProd;
  if (sym=="ReduceSum" ):
    return dc.opReduceSum;
  if (sym=="ReduceSumSquare" ):
    return dc.opReduceSumSquare;
  if (sym=="Relu" ):
    return dc.opRelu;
  if (sym=="Reshape" ):
    return dc.opReshape;
  if (sym=="Resize" ):
    return dc.opResize;
  if (sym=="ReverseSequence" ):
    return dc.opReverseSequence;
  if (sym=="RoiAlign" ):
    return dc.opRoiAlign;
  if (sym=="Round" ):
    return dc.opRound;
  if (sym=="Scan" ):
    return dc.opScan;
  if (sym=="Scatter" ):
    return dc.opScatter;
  if (sym=="Selu" ):
    return dc.opSelu;
  if (sym=="Shape" ):
    return dc.opShape;
  if (sym=="Shrink" ):
    return dc.opShrink;
  if (sym=="Sigmoid" ):
    return dc.opSigmoid;
  if (sym=="Sign" ):
    return dc.opSign;
  if (sym=="Sin" ):
    return dc.opSin;
  if (sym=="Sinh" ):
    return dc.opSinh;
  if (sym=="Size" ):
    return dc.opSize;
  if (sym=="Slice" ):
    return dc.opSlice;
  if (sym=="Softmax" ):
    return dc.opSoftmax;
  if (sym=="Softplus" ):
    return dc.opSoftplus;
  if (sym=="Softsign" ):
    return dc.opSoftsign;
  if (sym=="SpaceToDepth" ):
    return dc.opSpaceToDepth;
  if (sym=="Split" ):
    return dc.opSplit;
  if (sym=="Sqrt" ):
    return dc.opSqrt;
  if (sym=="Squeeze" ):
    return dc.opSqueeze;
  if (sym=="StringNormalizer" ):
    return dc.opStringNormalizer;
  if (sym=="Sub" ):
    return dc.opSub;
  if (sym=="Sum" ):
    return dc.opSum;
  if (sym=="Tan" ):
    return dc.opTan;
  if (sym=="Tanh" ):
    return dc.opTanh;
  if (sym=="TfIdfVectorizer" ):
    return dc.opTfIdfVectorizer;
  if (sym=="ThresholdedRelu" ):
    return dc.opThresholdedRelu;
  if (sym=="Tile" ):
    return dc.opTile;
  if (sym=="TopK" ):
    return dc.opTopK;
  if (sym=="Transpose" ):
    return dc.opTranspose;
  if (sym=="TrueDiv" ):
    return dc.opTrueDiv;
  if (sym=="Unsqueeze" ):
    return dc.opUnsqueeze;
  if (sym=="Upsample" ):
    return dc.opUpsample;
  if (sym=="Where" ):
    return dc.opWhere;
  if (sym=="Xor" ):
    return dc.opXor;
  return dc.opInvalid;

def dnncGraphNodeAttrCode(attr_str):
  if (attr_str=="activation_alpha" ):
    return dc.attr_activation_alpha;
  if (attr_str=="activation_beta" ):
    return dc.attr_activation_beta;
  if (attr_str=="activations" ):
    return dc.attr_activations;
  if (attr_str=="alpha" ):
    return dc.attr_alpha;
  if (attr_str=="auto_pad" ):
    return dc.attr_auto_pad;
  if (attr_str=="axes" ):
    return dc.attr_axes;
  if (attr_str=="axis" ):
    return dc.attr_axis;
  if (attr_str=="batch_axis" ):
    return dc.attr_batch_axis;
  if (attr_str=="beta" ):
    return dc.attr_beta;
  if (attr_str=="bias" ):
    return dc.attr_bias;
  if (attr_str=="blocksize" ):
    return dc.attr_blocksize;
  if (attr_str=="body" ):
    return dc.attr_body;
  if (attr_str=="case_change_action" ):
    return dc.attr_case_change_action;
  if (attr_str=="ceil_mode" ):
    return dc.attr_ceil_mode;
  if (attr_str=="center_point_box" ):
    return dc.attr_center_point_box;
  if (attr_str=="clip" ):
    return dc.attr_clip;
  if (attr_str=="count_include_pad" ):
    return dc.attr_count_include_pad;
  if (attr_str=="detect_negative" ):
    return dc.attr_detect_negative;
  if (attr_str=="detect_positive" ):
    return dc.attr_detect_positive;
  if (attr_str=="dilations" ):
    return dc.attr_dilations;
  if (attr_str=="direction" ):
    return dc.attr_direction;
  if (attr_str=="dtype" ):
    return dc.attr_dtype;
  if (attr_str=="else_branch" ):
    return dc.attr_else_branch;
  if (attr_str=="epsilon" ):
    return dc.attr_epsilon;
  if (attr_str=="exclusive" ):
    return dc.attr_exclusive;
  if (attr_str=="fmod" ):
    return dc.attr_fmod;
  if (attr_str=="gamma" ):
    return dc.attr_gamma;
  if (attr_str=="group" ):
    return dc.attr_group;
  if (attr_str=="hidden_size" ):
    return dc.attr_hidden_size;
  if (attr_str=="high" ):
    return dc.attr_high;
  if (attr_str=="input_forget" ):
    return dc.attr_input_forget;
  if (attr_str=="is_case_sensitive" ):
    return dc.attr_is_case_sensitive;
  if (attr_str=="k" ):
    return dc.attr_k;
  if (attr_str=="keepdims" ):
    return dc.attr_keepdims;
  if (attr_str=="kernel_shape" ):
    return dc.attr_kernel_shape;
  if (attr_str=="lambd" ):
    return dc.attr_lambd;
  if (attr_str=="larges" ):
    return dc.attr_larges;
  if (attr_str=="linear_before_reset" ):
    return dc.attr_linear_before_reset;
  if (attr_str=="locale" ):
    return dc.attr_locale;
  if (attr_str=="low" ):
    return dc.attr_low;
  if (attr_str=="max_gram_length" ):
    return dc.attr_max_gram_length;
  if (attr_str=="max_skip_count" ):
    return dc.attr_max_skip_count;
  if (attr_str=="mean" ):
    return dc.attr_mean;
  if (attr_str=="min_gram_length" ):
    return dc.attr_min_gram_length;
  if (attr_str=="mode" ):
    return dc.attr_mode;
  if (attr_str=="momentum" ):
    return dc.attr_momentum;
  if (attr_str=="ngram_counts" ):
    return dc.attr_ngram_counts;
  if (attr_str=="ngram_indexes" ):
    return dc.attr_ngram_indexes;
  if (attr_str=="num_scan_inputs" ):
    return dc.attr_num_scan_inputs;
  if (attr_str=="output_height" ):
    return dc.attr_output_height;
  if (attr_str=="output_padding" ):
    return dc.attr_output_padding;
  if (attr_str=="output_shape" ):
    return dc.attr_output_shape;
  if (attr_str=="output_width" ):
    return dc.attr_output_width;
  if (attr_str=="p" ):
    return dc.attr_p;
  if (attr_str=="pads" ):
    return dc.attr_pads;
  if (attr_str=="perm" ):
    return dc.attr_perm;
  if (attr_str=="pool_int64s" ):
    return dc.attr_pool_int64s;
  if (attr_str=="pool_strings" ):
    return dc.attr_pool_strings;
  if (attr_str=="pooled_shape" ):
    return dc.attr_pooled_shape;
  if (attr_str=="ratio" ):
    return dc.attr_ratio;
  if (attr_str=="reverse" ):
    return dc.attr_reverse;
  if (attr_str=="sample_size" ):
    return dc.attr_sample_size;
  if (attr_str=="sampling_ratio" ):
    return dc.attr_sampling_ratio;
  if (attr_str=="scale" ):
    return dc.attr_scale;
  if (attr_str=="scan_input_axes" ):
    return dc.attr_scan_input_axes;
  if (attr_str=="scan_input_directions" ):
    return dc.attr_scan_input_directions;
  if (attr_str=="scan_output_axes" ):
    return dc.attr_scan_output_axes;
  if (attr_str=="scan_output_directions" ):
    return dc.attr_scan_output_directions;
  if (attr_str=="seed" ):
    return dc.attr_seed;
  if (attr_str=="shape" ):
    return dc.attr_shape;
  if (attr_str=="size" ):
    return dc.attr_size;
  if (attr_str=="sorted" ):
    return dc.attr_sorted;
  if (attr_str=="spatial_scale" ):
    return dc.attr_spatial_scale;
  if (attr_str=="split" ):
    return dc.attr_split;
  if (attr_str=="stopwords" ):
    return dc.attr_stopwords;
  if (attr_str=="storage_order" ):
    return dc.attr_storage_order;
  if (attr_str=="strides" ):
    return dc.attr_strides;
  if (attr_str=="then_branch" ):
    return dc.attr_then_branch;
  if (attr_str=="time_axis" ):
    return dc.attr_time_axis;
  if (attr_str=="to" ):
    return dc.attr_to;
  if (attr_str=="transA" ):
    return dc.attr_transA;
  if (attr_str=="transB" ):
    return dc.attr_transB;
  if (attr_str=="value" ):
    return dc.attr_value;
  if (attr_str=="weights" ):
    return dc.attr_weights;
  return dc.attr_invalid;

class pbReader :
  """Reader class for DNNC models in ONNX binary/protobuf format."""

  def __init__(self):
      self._dcGraph = None ;

  def __del__(self):
      del self._dcGraph ;

  def createOPNode(self, node):

    op_type = dnncOpCode(node.op_type);
    if ( op_type is dc.opInvalid ):
      print("ERROR (ONNX):" +  node.op_type +" is not a valid graph-node op type.")
      return None

    dcNode = dc.node(op_type, node.name);

    for nd in node.input:
      dcNode.addInput(nd)

    for nd in node.output:
      dcNode.addOutput(nd)

    for attr in node.attribute:
      attr_type = dc.IR_DataType_NOTYPE;
      attr_vals = []
      attr_vec  = None
      if attr.type == onnx.AttributeProto.INT:
        attr_type = dc.IR_DataType_INT64;
        attr_vals.append(attr.i)
        attr_vec = dc.vectorInt(attr_vals)
      elif attr.type == onnx.AttributeProto.INTS:
        attr_type = dc.IR_DataType_INT64;
        for val in attr.ints:
          attr_vals.append(int(val))
        attr_vec = dc.vectorInt(attr_vals)
      elif attr.type == onnx.AttributeProto.FLOAT:
        attr_type = dc.IR_DataType_FLOAT;
        attr_vals.append(attr.f)
        attr_vec = dc.vectorFloat(attr_vals)
      elif attr.type == onnx.AttributeProto.FLOATS:
        attr_type = dc.IR_DataType_FLOAT;
        attr_vals.append(attr.floats)
        attr_vec = dc.vectorFloat(attr_vals)
      elif attr.type == onnx.AttributeProto.STRING:
        attr_type = dc.IR_DataType_STRING;
        attr_vals.append(str(attr.s))
        attr_vec = dc.vectorStr(attr_vals)
      elif attr.type == onnx.AttributeProto.STRINGS:
        attr_type = dc.IR_DataType_STRING;
        attr_vals.append(str(attr.strings))
        attr_vec = dc.vectorStr(attr_vals)
      elif attr.type == onnx.AttributeProto.TENSOR:
        if ( attr.t.data_type == dc.IR_DataType_INT8  or
             attr.t.data_type == dc.IR_DataType_INT16 or
             attr.t.data_type == dc.IR_DataType_INT32 or
             attr.t.data_type == dc.IR_DataType_INT64   ) :
          attr_type = attr.t.data_type
          pack_format = 'P';
          if ( attr.t.data_type == dc.IR_DataType_INT8 ) :
              pack_format = 'b'
          if ( attr.t.data_type == dc.IR_DataType_INT16) :
              pack_format = 'h'
          if ( attr.t.data_type == dc.IR_DataType_INT32) :
              pack_format = 'i'
          if ( attr.t.data_type == dc.IR_DataType_INT64) :
              pack_format = 'q'

          len=1
          for d in attr.t.dims:
            len *= d
          attr_data = struct.unpack(pack_format*len, attr.t.raw_data) ;
          attr_tensor = dc.iTensor(attr.t.dims, attr.name)
          attr_tensor.load(attr_data);
          attr_vec = dc.vectorTensorInt()
          attr_vec.push_back(attr_tensor)
        elif ( attr.t.data_type == dc.IR_DataType_FLOAT16 or
             attr.t.data_type == dc.IR_DataType_FLOAT   or
             attr.t.data_type == dc.IR_DataType_DOUBLE    ):
          attr_type = attr.t.data_type
          pack_format = 'P';
          if ( attr.t.data_type == dc.IR_DataType_FLOAT16 ) :
              pack_format = 'e'
          if ( attr.t.data_type == dc.IR_DataType_FLOAT ) :
              pack_format = 'f'
          if ( attr.t.data_type == dc.IR_DataType_DOUBLE ) :
              pack_format = 'd'
          len=1
          for d in attr.t.dims:
            len *= d
          attr_data = struct.unpack(pack_format*len, attr.t.raw_data) ;
          attr_tensor = dc.fTensor(attr.t.dims, attr.name)
          attr_tensor.load(attr_data);
          attr_vec = dc.vectorTensorFloat()
          attr_vec.push_back(attr_tensor)
        else:
          print("ERROR (ONNX): attribute tensor's datatype " + str(attr.t.data_type) +
                  " isn't understood.")

      elif attr.type == onnx.AttributeProto.TENSORS:
        attr_type = dc.IR_DataType_TENSORS;
        attr_vals.append(attr.tensors)
        attr_vec = dc.vectorTensorFloat(dc.fTensor(attr_vals))
      elif attr.type == onnx.AttributeProto.GRAPH:
        attr_type = dc.GRAPH;
        attr_vals.append(attr.g)
        print("ERROR (ONNX): sub-graph in graph-node is not yet supported.")
      elif attr.type == onnx.AttributeProto.GRAPHS:
        attr_type = dc.GRAPHS;
        attr_vals.append(attr.graphs)
        print("ERROR (ONNX): sub-graph in graph-node is not yet supported.")
      else:
        print("ERROR (ONNX): graph-node " + node.name + "\'s attribute " + \
               attr.name + " type " + str(attr.type) + " is not valid.")
        continue

      if ( attr_type is dc.IR_DataType_NOTYPE or attr_vec is None ) :
        continue ;

      attr_code = dnncGraphNodeAttrCode(attr.name)
      if ( attr_code is dc.attr_invalid ):
        print("WARN (ONNX): " + attr.name + " is not a valid graph-node attribute.")
        print("             operator " + node.op_type + " will be added without this attribute." )

      cAttrData = dc.genericData(attr_type,attr_vec) ;
      cAttr = dc.nodeAttribute(attr_code, cAttrData);
      dcNode.addAttribute(cAttr);


    return dcNode;

  def createTermNode(self, term):
    term_name  = term.name
    data_type  = dc.NOTYPE
    term_shape = []
    if ( term.type.tensor_type.elem_type ) :
      data_type  = term.type.tensor_type.elem_type
      if ( data_type <= dc.NOTYPE and data_type >= dc.TENSOR ) :
        print("ERROR (ONNX):  Term " + term_name + "\'s type " + data_type + " is not valid"  ) ;
        return ;

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

    return dc.placeHolder(term_name, data_type, term_shape)

  def main(self, onnx_filename):

    print("reading onnx model from file ", onnx_filename)

    model = onnx.load(onnx_filename)

    print("Model info:\n  ir_vesion : ", model.ir_version, "\n  doc       :", model.doc_string)
    graph = model.graph

    self._dcGraph = dc.Graph();
    self._dcGraph.setName(graph.name)

    nodes = graph.node
    for node in nodes:
      dcNode = self.createOPNode(node);
      if ( dcNode != None ):
        self._dcGraph.addNode(dcNode);

    for terminal in graph.input:
      dcTerm = self.createTermNode(terminal);
      if ( dcTerm != None ):
        self._dcGraph.addInput(dcTerm);

    for terminal in graph.output:
      dcTerm = self.createTermNode(terminal);
      if ( dcTerm != None ):
        self._dcGraph.addOutput(dcTerm);

    #for param in graph.initializer:
    #  dcParam = self.createParamNode(param);



if __name__ == "__main__":

  DNNC_ROOT=os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
  sys.path.append(DNNC_ROOT)
  from swig import dnnc as dc

  if len(sys.argv) >= 2:
    parser = pbReader()
    parser.main(sys.argv[1])
  else:
    print("\nUsage: "+sys.argv[0]+ " <onnx_model_file>.onnx \n")
