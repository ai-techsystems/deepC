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
if __name__ == "__main__":
  DNNC_PATH=os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..'+os.path.sep+'swig')
  sys.path.append(DNNC_PATH)

import onnx
import struct
import dnnc


def dnncOpCode(sym):
  if (sym=="Abs" ):
    return dnnc.opAbs;
  if (sym=="Acos" ):
    return dnnc.opAcos;
  if (sym=="Acosh" ):
    return dnnc.opAcosh;
  if (sym=="Add" ):
    return dnnc.opAdd;
  if (sym=="And" ):
    return dnnc.opAnd;
  if (sym=="ArgMax" ):
    return dnnc.opArgMax;
  if (sym=="ArgMin" ):
    return dnnc.opArgMin;
  if (sym=="Asin" ):
    return dnnc.opAsin;
  if (sym=="Asinh" ):
    return dnnc.opAsinh;
  if (sym=="Atan" ):
    return dnnc.opAtan;
  if (sym=="Atanh" ):
    return dnnc.opAtanh;
  if (sym=="AveragePool" ):
    return dnnc.opAveragePool;
  if (sym=="BatchNormalization" ):
    return dnnc.opBatchNormalization;
  if (sym=="BitShift" ):
    return dnnc.opBitShift;
  if (sym=="Cast" ):
    return dnnc.opCast;
  if (sym=="Ceil" ):
    return dnnc.opCeil;
  if (sym=="Clip" ):
    return dnnc.opClip;
  if (sym=="Compress" ):
    return dnnc.opCompress;
  if (sym=="Concat" ):
    return dnnc.opConcat;
  if (sym=="Constant" ):
    return dnnc.opConstant;
  if (sym=="ConstantOfShape" ):
    return dnnc.opConstantOfShape;
  if (sym=="Conv" ):
    return dnnc.opConv;
  if (sym=="ConvInteger" ):
    return dnnc.opConvInteger;
  if (sym=="ConvTranspose" ):
    return dnnc.opConvTranspose;
  if (sym=="Cos" ):
    return dnnc.opCos;
  if (sym=="Cosh" ):
    return dnnc.opCosh;
  if (sym=="CumSum" ):
    return dnnc.opCumSum;
  if (sym=="DepthToSpace" ):
    return dnnc.opDepthToSpace;
  if (sym=="DequantizeLinear" ):
    return dnnc.opDequantizeLinear;
  if (sym=="Div" ):
    return dnnc.opDiv;
  if (sym=="Dropout" ):
    return dnnc.opDropout;
  if (sym=="Elu" ):
    return dnnc.opElu;
  if (sym=="Equal" ):
    return dnnc.opEqual;
  if (sym=="Erf" ):
    return dnnc.opErf;
  if (sym=="Exp" ):
    return dnnc.opExp;
  if (sym=="Expand" ):
    return dnnc.opExpand;
  if (sym=="EyeLike" ):
    return dnnc.opEyeLike;
  if (sym=="Flatten" ):
    return dnnc.opFlatten;
  if (sym=="Floor" ):
    return dnnc.opFloor;
  if (sym=="FloorDiv" ):
    return dnnc.opFloorDiv;
  if (sym=="GRU" ):
    return dnnc.opGRU;
  if (sym=="Gather" ):
    return dnnc.opGather;
  if (sym=="Gemm" ):
    return dnnc.opGemm;
  if (sym=="GlobalAveragePool" ):
    return dnnc.opGlobalAveragePool;
  if (sym=="GlobalLpPool" ):
    return dnnc.opGlobalLpPool;
  if (sym=="GlobalMaxPool" ):
    return dnnc.opGlobalMaxPool;
  if (sym=="Greater" ):
    return dnnc.opGreater;
  if (sym=="GreaterEqual" ):
    return dnnc.opGreaterEqual;
  if (sym=="HardSigmoid" ):
    return dnnc.opHardSigmoid;
  if (sym=="Hardmax" ):
    return dnnc.opHardmax;
  if (sym=="Identity" ):
    return dnnc.opIdentity;
  if (sym=="If" ):
    return dnnc.opIf;
  if (sym=="InstanceNormalization" ):
    return dnnc.opInstanceNormalization;
  if (sym=="IsInf" ):
    return dnnc.opIsInf;
  if (sym=="IsNaN" ):
    return dnnc.opIsNaN;
  if (sym=="LRN" ):
    return dnnc.opLRN;
  if (sym=="LSTM" ):
    return dnnc.opLSTM;
  if (sym=="LeakyRelu" ):
    return dnnc.opLeakyRelu;
  if (sym=="Less" ):
    return dnnc.opLess;
  if (sym=="LessEqual" ):
    return dnnc.opLessEqual;
  if (sym=="Log" ):
    return dnnc.opLog;
  if (sym=="LogSoftmax" ):
    return dnnc.opLogSoftmax;
  if (sym=="Loop" ):
    return dnnc.opLoop;
  if (sym=="LpNormalization" ):
    return dnnc.opLpNormalization;
  if (sym=="LpPool" ):
    return dnnc.opLpPool;
  if (sym=="MatMul" ):
    return dnnc.opMatMul;
  if (sym=="MatMulInteger" ):
    return dnnc.opMatMulInteger;
  if (sym=="Max" ):
    return dnnc.opMax;
  if (sym=="MaxPool" ):
    return dnnc.opMaxPool;
  if (sym=="MaxRoiPool" ):
    return dnnc.opMaxRoiPool;
  if (sym=="MaxUnpool" ):
    return dnnc.opMaxUnpool;
  if (sym=="Mean" ):
    return dnnc.opMean;
  if (sym=="Min" ):
    return dnnc.opMin;
  if (sym=="Mod" ):
    return dnnc.opMod;
  if (sym=="Mul" ):
    return dnnc.opMul;
  if (sym=="Multinomial" ):
    return dnnc.opMultinomial;
  if (sym=="Neg" ):
    return dnnc.opNeg;
  if (sym=="NonMaxSuppression" ):
    return dnnc.opNonMaxSuppression;
  if (sym=="NonZero" ):
    return dnnc.opNonZero;
  if (sym=="Not" ):
    return dnnc.opNot;
  if (sym=="NotEqual" ):
    return dnnc.opNotEqual;
  if (sym=="OneHot" ):
    return dnnc.opOneHot;
  if (sym=="Or" ):
    return dnnc.opOr;
  if (sym=="PRelu" ):
    return dnnc.opPRelu;
  if (sym=="Pad" ):
    return dnnc.opPad;
  if (sym=="Pow" ):
    return dnnc.opPow;
  if (sym=="QLinearConv" ):
    return dnnc.opQLinearConv;
  if (sym=="QLinearMatMul" ):
    return dnnc.opQLinearMatMul;
  if (sym=="QuantizeLinear" ):
    return dnnc.opQuantizeLinear;
  if (sym=="RNN" ):
    return dnnc.opRNN;
  if (sym=="RandomNormal" ):
    return dnnc.opRandomNormal;
  if (sym=="RandomNormalLike" ):
    return dnnc.opRandomNormalLike;
  if (sym=="RandomUniform" ):
    return dnnc.opRandomUniform;
  if (sym=="RandomUniformLike" ):
    return dnnc.opRandomUniformLike;
  if (sym=="Reciprocal" ):
    return dnnc.opReciprocal;
  if (sym=="ReduceL1" ):
    return dnnc.opReduceL1;
  if (sym=="ReduceL2" ):
    return dnnc.opReduceL2;
  if (sym=="ReduceLogSum" ):
    return dnnc.opReduceLogSum;
  if (sym=="ReduceLogSumExp" ):
    return dnnc.opReduceLogSumExp;
  if (sym=="ReduceMax" ):
    return dnnc.opReduceMax;
  if (sym=="ReduceMean" ):
    return dnnc.opReduceMean;
  if (sym=="ReduceMin" ):
    return dnnc.opReduceMin;
  if (sym=="ReduceProd" ):
    return dnnc.opReduceProd;
  if (sym=="ReduceSum" ):
    return dnnc.opReduceSum;
  if (sym=="ReduceSumSquare" ):
    return dnnc.opReduceSumSquare;
  if (sym=="Relu" ):
    return dnnc.opRelu;
  if (sym=="Reshape" ):
    return dnnc.opReshape;
  if (sym=="Resize" ):
    return dnnc.opResize;
  if (sym=="ReverseSequence" ):
    return dnnc.opReverseSequence;
  if (sym=="RoiAlign" ):
    return dnnc.opRoiAlign;
  if (sym=="Round" ):
    return dnnc.opRound;
  if (sym=="Scan" ):
    return dnnc.opScan;
  if (sym=="Scatter" ):
    return dnnc.opScatter;
  if (sym=="Selu" ):
    return dnnc.opSelu;
  if (sym=="Shape" ):
    return dnnc.opShape;
  if (sym=="Shrink" ):
    return dnnc.opShrink;
  if (sym=="Sigmoid" ):
    return dnnc.opSigmoid;
  if (sym=="Sign" ):
    return dnnc.opSign;
  if (sym=="Sin" ):
    return dnnc.opSin;
  if (sym=="Sinh" ):
    return dnnc.opSinh;
  if (sym=="Size" ):
    return dnnc.opSize;
  if (sym=="Slice" ):
    return dnnc.opSlice;
  if (sym=="Softmax" ):
    return dnnc.opSoftmax;
  if (sym=="Softplus" ):
    return dnnc.opSoftplus;
  if (sym=="Softsign" ):
    return dnnc.opSoftsign;
  if (sym=="SpaceToDepth" ):
    return dnnc.opSpaceToDepth;
  if (sym=="Split" ):
    return dnnc.opSplit;
  if (sym=="Sqrt" ):
    return dnnc.opSqrt;
  if (sym=="Squeeze" ):
    return dnnc.opSqueeze;
  if (sym=="StringNormalizer" ):
    return dnnc.opStringNormalizer;
  if (sym=="Sub" ):
    return dnnc.opSub;
  if (sym=="Sum" ):
    return dnnc.opSum;
  if (sym=="Tan" ):
    return dnnc.opTan;
  if (sym=="Tanh" ):
    return dnnc.opTanh;
  if (sym=="TfIdfVectorizer" ):
    return dnnc.opTfIdfVectorizer;
  if (sym=="ThresholdedRelu" ):
    return dnnc.opThresholdedRelu;
  if (sym=="Tile" ):
    return dnnc.opTile;
  if (sym=="TopK" ):
    return dnnc.opTopK;
  if (sym=="Transpose" ):
    return dnnc.opTranspose;
  if (sym=="TrueDiv" ):
    return dnnc.opTrueDiv;
  if (sym=="Unsqueeze" ):
    return dnnc.opUnsqueeze;
  if (sym=="Upsample" ):
    return dnnc.opUpsample;
  if (sym=="Where" ):
    return dnnc.opWhere;
  if (sym=="Xor" ):
    return dnnc.opXor;
  return dnnc.opInvalid;

def dnncGraphNodeAttrCode(attr_str):
  if (attr_str=="activation_alpha" ):
    return dnnc.attr_activation_alpha;
  if (attr_str=="activation_beta" ):
    return dnnc.attr_activation_beta;
  if (attr_str=="activations" ):
    return dnnc.attr_activations;
  if (attr_str=="alpha" ):
    return dnnc.attr_alpha;
  if (attr_str=="auto_pad" ):
    return dnnc.attr_auto_pad;
  if (attr_str=="axes" ):
    return dnnc.attr_axes;
  if (attr_str=="axis" ):
    return dnnc.attr_axis;
  if (attr_str=="batch_axis" ):
    return dnnc.attr_batch_axis;
  if (attr_str=="beta" ):
    return dnnc.attr_beta;
  if (attr_str=="bias" ):
    return dnnc.attr_bias;
  if (attr_str=="blocksize" ):
    return dnnc.attr_blocksize;
  if (attr_str=="body" ):
    return dnnc.attr_body;
  if (attr_str=="case_change_action" ):
    return dnnc.attr_case_change_action;
  if (attr_str=="ceil_mode" ):
    return dnnc.attr_ceil_mode;
  if (attr_str=="center_point_box" ):
    return dnnc.attr_center_point_box;
  if (attr_str=="clip" ):
    return dnnc.attr_clip;
  if (attr_str=="count_include_pad" ):
    return dnnc.attr_count_include_pad;
  if (attr_str=="detect_negative" ):
    return dnnc.attr_detect_negative;
  if (attr_str=="detect_positive" ):
    return dnnc.attr_detect_positive;
  if (attr_str=="dilations" ):
    return dnnc.attr_dilations;
  if (attr_str=="direction" ):
    return dnnc.attr_direction;
  if (attr_str=="dtype" ):
    return dnnc.attr_dtype;
  if (attr_str=="else_branch" ):
    return dnnc.attr_else_branch;
  if (attr_str=="epsilon" ):
    return dnnc.attr_epsilon;
  if (attr_str=="exclusive" ):
    return dnnc.attr_exclusive;
  if (attr_str=="fmod" ):
    return dnnc.attr_fmod;
  if (attr_str=="gamma" ):
    return dnnc.attr_gamma;
  if (attr_str=="group" ):
    return dnnc.attr_group;
  if (attr_str=="hidden_size" ):
    return dnnc.attr_hidden_size;
  if (attr_str=="high" ):
    return dnnc.attr_high;
  if (attr_str=="input_forget" ):
    return dnnc.attr_input_forget;
  if (attr_str=="is_case_sensitive" ):
    return dnnc.attr_is_case_sensitive;
  if (attr_str=="k" ):
    return dnnc.attr_k;
  if (attr_str=="keepdims" ):
    return dnnc.attr_keepdims;
  if (attr_str=="kernel_shape" ):
    return dnnc.attr_kernel_shape;
  if (attr_str=="lambd" ):
    return dnnc.attr_lambd;
  if (attr_str=="larges" ):
    return dnnc.attr_larges;
  if (attr_str=="linear_before_reset" ):
    return dnnc.attr_linear_before_reset;
  if (attr_str=="locale" ):
    return dnnc.attr_locale;
  if (attr_str=="low" ):
    return dnnc.attr_low;
  if (attr_str=="max_gram_length" ):
    return dnnc.attr_max_gram_length;
  if (attr_str=="max_skip_count" ):
    return dnnc.attr_max_skip_count;
  if (attr_str=="mean" ):
    return dnnc.attr_mean;
  if (attr_str=="min_gram_length" ):
    return dnnc.attr_min_gram_length;
  if (attr_str=="mode" ):
    return dnnc.attr_mode;
  if (attr_str=="momentum" ):
    return dnnc.attr_momentum;
  if (attr_str=="ngram_counts" ):
    return dnnc.attr_ngram_counts;
  if (attr_str=="ngram_indexes" ):
    return dnnc.attr_ngram_indexes;
  if (attr_str=="num_scan_inputs" ):
    return dnnc.attr_num_scan_inputs;
  if (attr_str=="output_height" ):
    return dnnc.attr_output_height;
  if (attr_str=="output_padding" ):
    return dnnc.attr_output_padding;
  if (attr_str=="output_shape" ):
    return dnnc.attr_output_shape;
  if (attr_str=="output_width" ):
    return dnnc.attr_output_width;
  if (attr_str=="p" ):
    return dnnc.attr_p;
  if (attr_str=="pads" ):
    return dnnc.attr_pads;
  if (attr_str=="perm" ):
    return dnnc.attr_perm;
  if (attr_str=="pool_int64s" ):
    return dnnc.attr_pool_int64s;
  if (attr_str=="pool_strings" ):
    return dnnc.attr_pool_strings;
  if (attr_str=="pooled_shape" ):
    return dnnc.attr_pooled_shape;
  if (attr_str=="ratio" ):
    return dnnc.attr_ratio;
  if (attr_str=="reverse" ):
    return dnnc.attr_reverse;
  if (attr_str=="sample_size" ):
    return dnnc.attr_sample_size;
  if (attr_str=="sampling_ratio" ):
    return dnnc.attr_sampling_ratio;
  if (attr_str=="scale" ):
    return dnnc.attr_scale;
  if (attr_str=="scan_input_axes" ):
    return dnnc.attr_scan_input_axes;
  if (attr_str=="scan_input_directions" ):
    return dnnc.attr_scan_input_directions;
  if (attr_str=="scan_output_axes" ):
    return dnnc.attr_scan_output_axes;
  if (attr_str=="scan_output_directions" ):
    return dnnc.attr_scan_output_directions;
  if (attr_str=="seed" ):
    return dnnc.attr_seed;
  if (attr_str=="shape" ):
    return dnnc.attr_shape;
  if (attr_str=="size" ):
    return dnnc.attr_size;
  if (attr_str=="sorted" ):
    return dnnc.attr_sorted;
  if (attr_str=="spatial_scale" ):
    return dnnc.attr_spatial_scale;
  if (attr_str=="split" ):
    return dnnc.attr_split;
  if (attr_str=="stopwords" ):
    return dnnc.attr_stopwords;
  if (attr_str=="storage_order" ):
    return dnnc.attr_storage_order;
  if (attr_str=="strides" ):
    return dnnc.attr_strides;
  if (attr_str=="then_branch" ):
    return dnnc.attr_then_branch;
  if (attr_str=="time_axis" ):
    return dnnc.attr_time_axis;
  if (attr_str=="to" ):
    return dnnc.attr_to;
  if (attr_str=="transA" ):
    return dnnc.attr_transA;
  if (attr_str=="transB" ):
    return dnnc.attr_transB;
  if (attr_str=="value" ):
    return dnnc.attr_value;
  if (attr_str=="weights" ):
    return dnnc.attr_weights;
  return dnnc.attr_invalid;


class pbReader :
  """Reader class for DNNC models in ONNX binary/protobuf format."""

  def __init__(self):
      if sys.modules.get('dnnc') is None:
        print("ERROR (DNNC): could not find dnnc module. Please make sure dnnc is imported before calling ", __name__)
      self._dcGraph = None ;

  def __del__(self):
      del self._dcGraph ;

  def createOPNode(self, node):

    op_type = dnncOpCode(node.op_type);
    if ( op_type is dnnc.opInvalid ):
      print("ERROR (ONNX):" +  node.op_type +" is not a valid graph-node op type.")
      return None

    dcNode = dnnc.node(op_type, node.name);

    for nd in node.input:
      dcNode.addInput(nd)

    for nd in node.output:
      dcNode.addOutput(nd)

    for attr in node.attribute:
      attr_type = dnnc.IR_DataType_NOTYPE;
      attr_vals = []
      attr_vec  = None
      if attr.type == onnx.AttributeProto.INT:
        attr_type = dnnc.IR_DataType_INT64;
        attr_vals.append(attr.i)
        attr_vec = dnnc.vectorInt(attr_vals)
      elif attr.type == onnx.AttributeProto.INTS:
        attr_type = dnnc.IR_DataType_INT64;
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

      if ( attr_type is dnnc.IR_DataType_NOTYPE or attr_vec is None ) :
        continue ;

      attr_code = dnncGraphNodeAttrCode(attr.name)
      if ( attr_code is dnnc.attr_invalid ):
        print("WARN (ONNX): " + attr.name + " is not a valid graph-node attribute.")
        print("             operator " + node.op_type + " will be added without this attribute." )

      cAttrData = dnnc.genericData(attr_type,attr_vec) ;
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

    return dnnc.placeHolder(term_name, data_type, term_shape)

  def main(self, onnx_filename):
    if sys.modules.get('dnnc') is None:
      print("ERROR (DNNC): could not find dnnc module. Please make sure dnnc is imported before calling ", __name__)
      return ;

    print("reading onnx model from file ", onnx_filename)

    model = onnx.load(onnx_filename)

    print("Model info:\n  ir_vesion : ", model.ir_version, "\n  doc       :", model.doc_string)
    graph = model.graph

    self._dcGraph = dnnc.Graph();
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
    return self._dcGraph

if __name__ == "__main__":
  if len(sys.argv) >= 2:
    parser = pbReader()
    parser.main(sys.argv[1])
  else:
    print("\nUsage: "+sys.argv[0]+ " <onnx_model_file>.onnx \n")
