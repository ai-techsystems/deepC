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
#include "operators/opTypes.h"
namespace dnnc {

OPATTR getAttrName(std::string attrStr) {

  if (attrStr == "activation_alpha")
    return attr_activation_alpha;
  if (attrStr == "activation_beta")
    return attr_activation_beta;
  if (attrStr == "activations")
    return attr_activations;
  if (attrStr == "alpha")
    return attr_alpha;
  if (attrStr == "auto_pad")
    return attr_auto_pad;
  if (attrStr == "axes")
    return attr_axes;
  if (attrStr == "axis")
    return attr_axis;
  if (attrStr == "batch_axis")
    return attr_batch_axis;
  if (attrStr == "beta")
    return attr_beta;
  if (attrStr == "bias")
    return attr_bias;
  if (attrStr == "blocksize")
    return attr_blocksize;
  if (attrStr == "body")
    return attr_body;
  if (attrStr == "case_change_action")
    return attr_case_change_action;
  if (attrStr == "ceil_mode")
    return attr_ceil_mode;
  if (attrStr == "center_point_box")
    return attr_center_point_box;
  if (attrStr == "clip")
    return attr_clip;
  if (attrStr == "count_include_pad")
    return attr_count_include_pad;
  if (attrStr == "detect_negative")
    return attr_detect_negative;
  if (attrStr == "detect_positive")
    return attr_detect_positive;
  if (attrStr == "dilations")
    return attr_dilations;
  if (attrStr == "direction")
    return attr_direction;
  if (attrStr == "dtype")
    return attr_dtype;
  if (attrStr == "else_branch")
    return attr_else_branch;
  if (attrStr == "epsilon")
    return attr_epsilon;
  if (attrStr == "exclusive")
    return attr_exclusive;
  if (attrStr == "fmod")
    return attr_fmod;
  if (attrStr == "gamma")
    return attr_gamma;
  if (attrStr == "group")
    return attr_group;
  if (attrStr == "hidden_size")
    return attr_hidden_size;
  if (attrStr == "high")
    return attr_high;
  if (attrStr == "input_forget")
    return attr_input_forget;
  if (attrStr == "is_case_sensitive")
    return attr_is_case_sensitive;
  if (attrStr == "k")
    return attr_k;
  if (attrStr == "keepdims")
    return attr_keepdims;
  if (attrStr == "kernel_shape")
    return attr_kernel_shape;
  if (attrStr == "lambd")
    return attr_lambd;
  if (attrStr == "larges")
    return attr_larges;
  if (attrStr == "linear_before_reset")
    return attr_linear_before_reset;
  if (attrStr == "locale")
    return attr_locale;
  if (attrStr == "low")
    return attr_low;
  if (attrStr == "max_gram_length")
    return attr_max_gram_length;
  if (attrStr == "max_skip_count")
    return attr_max_skip_count;
  if (attrStr == "mean")
    return attr_mean;
  if (attrStr == "min_gram_length")
    return attr_min_gram_length;
  if (attrStr == "mode")
    return attr_mode;
  if (attrStr == "momentum")
    return attr_momentum;
  if (attrStr == "ngram_counts")
    return attr_ngram_counts;
  if (attrStr == "ngram_indexes")
    return attr_ngram_indexes;
  if (attrStr == "num_scan_inputs")
    return attr_num_scan_inputs;
  if (attrStr == "output_height")
    return attr_output_height;
  if (attrStr == "output_padding")
    return attr_output_padding;
  if (attrStr == "output_shape")
    return attr_output_shape;
  if (attrStr == "output_width")
    return attr_output_width;
  if (attrStr == "p")
    return attr_p;
  if (attrStr == "pads")
    return attr_pads;
  if (attrStr == "perm")
    return attr_perm;
  if (attrStr == "pool_int64s")
    return attr_pool_int64s;
  if (attrStr == "pool_strings")
    return attr_pool_strings;
  if (attrStr == "pooled_shape")
    return attr_pooled_shape;
  if (attrStr == "ratio")
    return attr_ratio;
  if (attrStr == "reverse")
    return attr_reverse;
  if (attrStr == "sample_size")
    return attr_sample_size;
  if (attrStr == "sampling_ratio")
    return attr_sampling_ratio;
  if (attrStr == "scale")
    return attr_scale;
  if (attrStr == "scan_input_axes")
    return attr_scan_input_axes;
  if (attrStr == "scan_input_directions")
    return attr_scan_input_directions;
  if (attrStr == "scan_output_axes")
    return attr_scan_output_axes;
  if (attrStr == "scan_output_directions")
    return attr_scan_output_directions;
  if (attrStr == "seed")
    return attr_seed;
  if (attrStr == "shape")
    return attr_shape;
  if (attrStr == "size")
    return attr_size;
  if (attrStr == "sorted")
    return attr_sorted;
  if (attrStr == "spatial_scale")
    return attr_spatial_scale;
  if (attrStr == "split")
    return attr_split;
  if (attrStr == "stopwords")
    return attr_stopwords;
  if (attrStr == "storage_order")
    return attr_storage_order;
  if (attrStr == "strides")
    return attr_strides;
  if (attrStr == "then_branch")
    return attr_then_branch;
  if (attrStr == "time_axis")
    return attr_time_axis;
  if (attrStr == "to")
    return attr_to;
  if (attrStr == "transA")
    return attr_transA;
  if (attrStr == "transB")
    return attr_transB;
  if (attrStr == "value")
    return attr_value;
  if (attrStr == "weights")
    return attr_weights;
  if (attrStr == "invalid")
    return attr_invalid;

  return attr_invalid;
}

std::string getAttrNameStr(OPATTR attr) {

  switch (attr) {
  case (attr_activation_alpha):
    return "activation_alpha";
    break;
  case (attr_activation_beta):
    return "activation_beta";
    break;
  case (attr_activations):
    return "activations";
    break;
  case (attr_alpha):
    return "alpha";
    break;
  case (attr_auto_pad):
    return "auto_pad";
    break;
  case (attr_axes):
    return "axes";
    break;
  case (attr_axis):
    return "axis";
    break;
  case (attr_batch_axis):
    return "batch_axis";
    break;
  case (attr_beta):
    return "beta";
    break;
  case (attr_bias):
    return "bias";
    break;
  case (attr_blocksize):
    return "blocksize";
    break;
  case (attr_body):
    return "body";
    break;
  case (attr_case_change_action):
    return "case_change_action";
    break;
  case (attr_ceil_mode):
    return "ceil_mode";
    break;
  case (attr_center_point_box):
    return "center_point_box";
    break;
  case (attr_clip):
    return "clip";
    break;
  case (attr_count_include_pad):
    return "count_include_pad";
    break;
  case (attr_detect_negative):
    return "detect_negative";
    break;
  case (attr_detect_positive):
    return "detect_positive";
    break;
  case (attr_dilations):
    return "dilations";
    break;
  case (attr_direction):
    return "direction";
    break;
  case (attr_dtype):
    return "dtype";
    break;
  case (attr_else_branch):
    return "else_branch";
    break;
  case (attr_epsilon):
    return "epsilon";
    break;
  case (attr_exclusive):
    return "exclusive";
    break;
  case (attr_fmod):
    return "fmod";
    break;
  case (attr_gamma):
    return "gamma";
    break;
  case (attr_group):
    return "group";
    break;
  case (attr_hidden_size):
    return "hidden_size";
    break;
  case (attr_high):
    return "high";
    break;
  case (attr_input_forget):
    return "input_forget";
    break;
  case (attr_is_case_sensitive):
    return "is_case_sensitive";
    break;
  case (attr_k):
    return "k";
    break;
  case (attr_keepdims):
    return "keepdims";
    break;
  case (attr_kernel_shape):
    return "kernel_shape";
    break;
  case (attr_lambd):
    return "lambd";
    break;
  case (attr_larges):
    return "larges";
    break;
  case (attr_linear_before_reset):
    return "linear_before_reset";
    break;
  case (attr_locale):
    return "locale";
    break;
  case (attr_low):
    return "low";
    break;
  case (attr_max_gram_length):
    return "max_gram_length";
    break;
  case (attr_max_skip_count):
    return "max_skip_count";
    break;
  case (attr_mean):
    return "mean";
    break;
  case (attr_min_gram_length):
    return "min_gram_length";
    break;
  case (attr_mode):
    return "mode";
    break;
  case (attr_momentum):
    return "momentum";
    break;
  case (attr_ngram_counts):
    return "ngram_counts";
    break;
  case (attr_ngram_indexes):
    return "ngram_indexes";
    break;
  case (attr_num_scan_inputs):
    return "num_scan_inputs";
    break;
  case (attr_output_height):
    return "output_height";
    break;
  case (attr_output_padding):
    return "output_padding";
    break;
  case (attr_output_shape):
    return "output_shape";
    break;
  case (attr_output_width):
    return "output_width";
    break;
  case (attr_p):
    return "p";
    break;
  case (attr_pads):
    return "pads";
    break;
  case (attr_perm):
    return "perm";
    break;
  case (attr_pool_int64s):
    return "pool_int64s";
    break;
  case (attr_pool_strings):
    return "pool_strings";
    break;
  case (attr_pooled_shape):
    return "pooled_shape";
    break;
  case (attr_ratio):
    return "ratio";
    break;
  case (attr_reverse):
    return "reverse";
    break;
  case (attr_sample_size):
    return "sample_size";
    break;
  case (attr_sampling_ratio):
    return "sampling_ratio";
    break;
  case (attr_scale):
    return "scale";
    break;
  case (attr_scan_input_axes):
    return "scan_input_axes";
    break;
  case (attr_scan_input_directions):
    return "scan_input_directions";
    break;
  case (attr_scan_output_axes):
    return "scan_output_axes";
    break;
  case (attr_scan_output_directions):
    return "scan_output_directions";
    break;
  case (attr_seed):
    return "seed";
    break;
  case (attr_shape):
    return "shape";
    break;
  case (attr_size):
    return "size";
    break;
  case (attr_sorted):
    return "sorted";
    break;
  case (attr_spatial_scale):
    return "spatial_scale";
    break;
  case (attr_split):
    return "split";
    break;
  case (attr_stopwords):
    return "stopwords";
    break;
  case (attr_storage_order):
    return "storage_order";
    break;
  case (attr_strides):
    return "strides";
    break;
  case (attr_then_branch):
    return "then_branch";
    break;
  case (attr_time_axis):
    return "time_axis";
    break;
  case (attr_to):
    return "to";
    break;
  case (attr_transA):
    return "transA";
    break;
  case (attr_transB):
    return "transB";
    break;
  case (attr_value):
    return "value";
    break;
  case (attr_weights):
    return "weights";
    break;
  case (attr_invalid):
  default:
    return "invalid";
    break;
  }
  return "invalid";
}

OPCODE getOpCode(std::string opCodeStr) {

  if (opCodeStr == "Abs")
    return opAbs;
  if (opCodeStr == "Acos")
    return opAcos;
  if (opCodeStr == "Acosh")
    return opAcosh;
  if (opCodeStr == "Add")
    return opAdd;
  if (opCodeStr == "And")
    return opAnd;
  if (opCodeStr == "ArgMax")
    return opArgMax;
  if (opCodeStr == "ArgMin")
    return opArgMin;
  if (opCodeStr == "Asin")
    return opAsin;
  if (opCodeStr == "Asinh")
    return opAsinh;
  if (opCodeStr == "Atan")
    return opAtan;
  if (opCodeStr == "Atanh")
    return opAtanh;
  if (opCodeStr == "AveragePool")
    return opAveragePool;
  if (opCodeStr == "BatchNormalization")
    return opBatchNormalization;
  if (opCodeStr == "BitShift")
    return opBitShift;
  if (opCodeStr == "BitwiseAnd")
    return opBitwiseAnd;
  if (opCodeStr == "BitwiseOr")
    return opBitwiseOr;
  if (opCodeStr == "BitwiseXor")
    return opBitwiseXor;
  if (opCodeStr == "Cast")
    return opCast;
  if (opCodeStr == "Ceil")
    return opCeil;
  if (opCodeStr == "Clip")
    return opClip;
  if (opCodeStr == "Compress")
    return opCompress;
  if (opCodeStr == "Concat")
    return opConcat;
  if (opCodeStr == "Constant")
    return opConstant;
  if (opCodeStr == "ConstantOfShape")
    return opConstantOfShape;
  if (opCodeStr == "Conv")
    return opConv;
  if (opCodeStr == "ConvInteger")
    return opConvInteger;
  if (opCodeStr == "ConvTranspose")
    return opConvTranspose;
  if (opCodeStr == "Cos")
    return opCos;
  if (opCodeStr == "Cosh")
    return opCosh;
  if (opCodeStr == "CumSum")
    return opCumSum;
  if (opCodeStr == "DepthToSpace")
    return opDepthToSpace;
  if (opCodeStr == "DequantizeLinear")
    return opDequantizeLinear;
  if (opCodeStr == "Div")
    return opDiv;
  if (opCodeStr == "Dropout")
    return opDropout;
  if (opCodeStr == "Elu")
    return opElu;
  if (opCodeStr == "Equal")
    return opEqual;
  if (opCodeStr == "Erf")
    return opErf;
  if (opCodeStr == "Exp")
    return opExp;
  if (opCodeStr == "Expand")
    return opExpand;
  if (opCodeStr == "EyeLike")
    return opEyeLike;
  if (opCodeStr == "Flatten")
    return opFlatten;
  if (opCodeStr == "Floor")
    return opFloor;
  if (opCodeStr == "FloorDiv")
    return opFloorDiv;
  if (opCodeStr == "GRU")
    return opGRU;
  if (opCodeStr == "Gather")
    return opGather;
  if (opCodeStr == "Gemm")
    return opGemm;
  if (opCodeStr == "GlobalAveragePool")
    return opGlobalAveragePool;
  if (opCodeStr == "GlobalLpPool")
    return opGlobalLpPool;
  if (opCodeStr == "GlobalMaxPool")
    return opGlobalMaxPool;
  if (opCodeStr == "Greater")
    return opGreater;
  if (opCodeStr == "GreaterEqual")
    return opGreaterEqual;
  if (opCodeStr == "HardSigmoid")
    return opHardSigmoid;
  if (opCodeStr == "Hardmax")
    return opHardmax;
  if (opCodeStr == "Identity")
    return opIdentity;
  if (opCodeStr == "If")
    return opIf;
  if (opCodeStr == "InstanceNormalization")
    return opInstanceNormalization;
  if (opCodeStr == "IsInf")
    return opIsInf;
  if (opCodeStr == "IsNaN")
    return opIsNaN;
  if (opCodeStr == "LRN")
    return opLRN;
  if (opCodeStr == "LSTM")
    return opLSTM;
  if (opCodeStr == "LeakyRelu")
    return opLeakyRelu;
  if (opCodeStr == "Less")
    return opLess;
  if (opCodeStr == "LessEqual")
    return opLessEqual;
  if (opCodeStr == "Log")
    return opLog;
  if (opCodeStr == "LogSoftmax")
    return opLogSoftmax;
  if (opCodeStr == "Loop")
    return opLoop;
  if (opCodeStr == "LpNormalization")
    return opLpNormalization;
  if (opCodeStr == "LpPool")
    return opLpPool;
  if (opCodeStr == "MatMul")
    return opMatMul;
  if (opCodeStr == "MatMulInteger")
    return opMatMulInteger;
  if (opCodeStr == "Max")
    return opMax;
  if (opCodeStr == "MaxPool")
    return opMaxPool;
  if (opCodeStr == "MaxRoiPool")
    return opMaxRoiPool;
  if (opCodeStr == "MaxUnpool")
    return opMaxUnpool;
  if (opCodeStr == "Mean")
    return opMean;
  if (opCodeStr == "Min")
    return opMin;
  if (opCodeStr == "Mod")
    return opMod;
  if (opCodeStr == "Mul")
    return opMul;
  if (opCodeStr == "Multinomial")
    return opMultinomial;
  if (opCodeStr == "Neg")
    return opNeg;
  if (opCodeStr == "NonMaxSuppression")
    return opNonMaxSuppression;
  if (opCodeStr == "NonZero")
    return opNonZero;
  if (opCodeStr == "Not")
    return opNot;
  if (opCodeStr == "NotEqual")
    return opNotEqual;
  if (opCodeStr == "OneHot")
    return opOneHot;
  if (opCodeStr == "Or")
    return opOr;
  if (opCodeStr == "PRelu")
    return opPRelu;
  if (opCodeStr == "Pad")
    return opPad;
  if (opCodeStr == "Pow")
    return opPow;
  if (opCodeStr == "QLinearConv")
    return opQLinearConv;
  if (opCodeStr == "QLinearMatMul")
    return opQLinearMatMul;
  if (opCodeStr == "QuantizeLinear")
    return opQuantizeLinear;
  if (opCodeStr == "RNN")
    return opRNN;
  if (opCodeStr == "RandomNormal")
    return opRandomNormal;
  if (opCodeStr == "RandomNormalLike")
    return opRandomNormalLike;
  if (opCodeStr == "RandomUniform")
    return opRandomUniform;
  if (opCodeStr == "RandomUniformLike")
    return opRandomUniformLike;
  if (opCodeStr == "Reciprocal")
    return opReciprocal;
  if (opCodeStr == "ReduceL1")
    return opReduceL1;
  if (opCodeStr == "ReduceL2")
    return opReduceL2;
  if (opCodeStr == "ReduceLogSum")
    return opReduceLogSum;
  if (opCodeStr == "ReduceLogSumExp")
    return opReduceLogSumExp;
  if (opCodeStr == "ReduceMax")
    return opReduceMax;
  if (opCodeStr == "ReduceMean")
    return opReduceMean;
  if (opCodeStr == "ReduceMin")
    return opReduceMin;
  if (opCodeStr == "ReduceProd")
    return opReduceProd;
  if (opCodeStr == "ReduceSum")
    return opReduceSum;
  if (opCodeStr == "ReduceSumSquare")
    return opReduceSumSquare;
  if (opCodeStr == "Relu")
    return opRelu;
  if (opCodeStr == "Remainder")
    return opRemainder;
  if (opCodeStr == "Reshape")
    return opReshape;
  if (opCodeStr == "Resize")
    return opResize;
  if (opCodeStr == "ReverseSequence")
    return opReverseSequence;
  if (opCodeStr == "RoiAlign")
    return opRoiAlign;
  if (opCodeStr == "Round")
    return opRound;
  if (opCodeStr == "Scan")
    return opScan;
  if (opCodeStr == "Scatter")
    return opScatter;
  if (opCodeStr == "Selu")
    return opSelu;
  if (opCodeStr == "Shape")
    return opShape;
  if (opCodeStr == "Shrink")
    return opShrink;
  if (opCodeStr == "Sigmoid")
    return opSigmoid;
  if (opCodeStr == "Sign")
    return opSign;
  if (opCodeStr == "Sin")
    return opSin;
  if (opCodeStr == "Sinh")
    return opSinh;
  if (opCodeStr == "Size")
    return opSize;
  if (opCodeStr == "Slice")
    return opSlice;
  if (opCodeStr == "Softmax")
    return opSoftmax;
  if (opCodeStr == "Softplus")
    return opSoftplus;
  if (opCodeStr == "Softsign")
    return opSoftsign;
  if (opCodeStr == "SpaceToDepth")
    return opSpaceToDepth;
  if (opCodeStr == "Split")
    return opSplit;
  if (opCodeStr == "Sqrt")
    return opSqrt;
  if (opCodeStr == "Squeeze")
    return opSqueeze;
  if (opCodeStr == "StringNormalizer")
    return opStringNormalizer;
  if (opCodeStr == "Sub")
    return opSub;
  if (opCodeStr == "Sum")
    return opSum;
  if (opCodeStr == "Tan")
    return opTan;
  if (opCodeStr == "Tanh")
    return opTanh;
  if (opCodeStr == "TfIdfVectorizer")
    return opTfIdfVectorizer;
  if (opCodeStr == "ThresholdedRelu")
    return opThresholdedRelu;
  if (opCodeStr == "Tile")
    return opTile;
  if (opCodeStr == "TopK")
    return opTopK;
  if (opCodeStr == "Transpose")
    return opTranspose;
  if (opCodeStr == "TrueDiv")
    return opTrueDiv;
  if (opCodeStr == "Unsqueeze")
    return opUnsqueeze;
  if (opCodeStr == "Upsample")
    return opUpsample;
  if (opCodeStr == "Where")
    return opWhere;
  if (opCodeStr == "Xor")
    return opXor;
  if (opCodeStr == "Invalid")
    return opInvalid;

  return opInvalid;
}

std::string getOpCodeStr(OPCODE opCode) {
  switch (opCode) {
  case (opAbs):
    return "Abs";
    break;
  case (opAcos):
    return "Acos";
    break;
  case (opAcosh):
    return "Acosh";
    break;
  case (opAdd):
    return "Add";
    break;
  case (opAnd):
    return "And";
    break;
  case (opArgMax):
    return "ArgMax";
    break;
  case (opArgMin):
    return "ArgMin";
    break;
  case (opAsin):
    return "Asin";
    break;
  case (opAsinh):
    return "Asinh";
    break;
  case (opAtan):
    return "Atan";
    break;
  case (opAtanh):
    return "Atanh";
    break;
  case (opAveragePool):
    return "AveragePool";
    break;
  case (opBatchNormalization):
    return "BatchNormalization";
    break;
  case (opBitShift):
    return "BitShift";
    break;
  case (opCast):
    return "Cast";
    break;
  case (opCeil):
    return "Ceil";
    break;
  case (opClip):
    return "Clip";
    break;
  case (opCompress):
    return "Compress";
    break;
  case (opConcat):
    return "Concat";
    break;
  case (opConstant):
    return "Constant";
    break;
  case (opConstantOfShape):
    return "ConstantOfShape";
    break;
  case (opConv):
    return "Conv";
    break;
  case (opConvInteger):
    return "ConvInteger";
    break;
  case (opConvTranspose):
    return "ConvTranspose";
    break;
  case (opCos):
    return "Cos";
    break;
  case (opCosh):
    return "Cosh";
    break;
  case (opCumSum):
    return "CumSum";
    break;
  case (opDepthToSpace):
    return "DepthToSpace";
    break;
  case (opDequantizeLinear):
    return "DequantizeLinear";
    break;
  case (opDiv):
    return "Div";
    break;
  case (opDropout):
    return "Dropout";
    break;
  case (opElu):
    return "Elu";
    break;
  case (opEqual):
    return "Equal";
    break;
  case (opErf):
    return "Erf";
    break;
  case (opExp):
    return "Exp";
    break;
  case (opExpand):
    return "Expand";
    break;
  case (opEyeLike):
    return "EyeLike";
    break;
  case (opFlatten):
    return "Flatten";
    break;
  case (opFloor):
    return "Floor";
    break;
  case (opFloorDiv):
    return "FloorDiv";
    break;
  case (opGRU):
    return "GRU";
    break;
  case (opGather):
    return "Gather";
    break;
  case (opGemm):
    return "Gemm";
    break;
  case (opGlobalAveragePool):
    return "GlobalAveragePool";
    break;
  case (opGlobalLpPool):
    return "GlobalLpPool";
    break;
  case (opGlobalMaxPool):
    return "GlobalMaxPool";
    break;
  case (opGreater):
    return "Greater";
    break;
  case (opGreaterEqual):
    return "GreaterEqual";
    break;
  case (opHardSigmoid):
    return "HardSigmoid";
    break;
  case (opHardmax):
    return "Hardmax";
    break;
  case (opIdentity):
    return "Identity";
    break;
  case (opIf):
    return "If";
    break;
  case (opInstanceNormalization):
    return "InstanceNormalization";
    break;
  case (opIsInf):
    return "IsInf";
    break;
  case (opIsNaN):
    return "IsNaN";
    break;
  case (opLRN):
    return "LRN";
    break;
  case (opLSTM):
    return "LSTM";
    break;
  case (opLeakyRelu):
    return "LeakyRelu";
    break;
  case (opLess):
    return "Less";
    break;
  case (opLessEqual):
    return "LessEqual";
    break;
  case (opLog):
    return "Log";
    break;
  case (opLogSoftmax):
    return "LogSoftmax";
    break;
  case (opLoop):
    return "Loop";
    break;
  case (opLpNormalization):
    return "LpNormalization";
    break;
  case (opLpPool):
    return "LpPool";
    break;
  case (opMatMul):
    return "MatMul";
    break;
  case (opMatMulInteger):
    return "MatMulInteger";
    break;
  case (opMax):
    return "Max";
    break;
  case (opMaxPool):
    return "MaxPool";
    break;
  case (opMaxRoiPool):
    return "MaxRoiPool";
    break;
  case (opMaxUnpool):
    return "MaxUnpool";
    break;
  case (opMean):
    return "Mean";
    break;
  case (opMin):
    return "Min";
    break;
  case (opMod):
    return "Mod";
    break;
  case (opMul):
    return "Mul";
    break;
  case (opMultinomial):
    return "Multinomial";
    break;
  case (opNeg):
    return "Neg";
    break;
  case (opNonMaxSuppression):
    return "NonMaxSuppression";
    break;
  case (opNonZero):
    return "NonZero";
    break;
  case (opNot):
    return "Not";
    break;
  case (opNotEqual):
    return "NotEqual";
    break;
  case (opOneHot):
    return "OneHot";
    break;
  case (opOr):
    return "Or";
    break;
  case (opPRelu):
    return "PRelu";
    break;
  case (opPad):
    return "Pad";
    break;
  case (opPow):
    return "Pow";
    break;
  case (opQLinearConv):
    return "QLinearConv";
    break;
  case (opQLinearMatMul):
    return "QLinearMatMul";
    break;
  case (opQuantizeLinear):
    return "QuantizeLinear";
    break;
  case (opRNN):
    return "RNN";
    break;
  case (opRandomNormal):
    return "RandomNormal";
    break;
  case (opRandomNormalLike):
    return "RandomNormalLike";
    break;
  case (opRandomUniform):
    return "RandomUniform";
    break;
  case (opRandomUniformLike):
    return "RandomUniformLike";
    break;
  case (opReciprocal):
    return "Reciprocal";
    break;
  case (opReduceL1):
    return "ReduceL1";
    break;
  case (opReduceL2):
    return "ReduceL2";
    break;
  case (opReduceLogSum):
    return "ReduceLogSum";
    break;
  case (opReduceLogSumExp):
    return "ReduceLogSumExp";
    break;
  case (opReduceMax):
    return "ReduceMax";
    break;
  case (opReduceMean):
    return "ReduceMean";
    break;
  case (opReduceMin):
    return "ReduceMin";
    break;
  case (opReduceProd):
    return "ReduceProd";
    break;
  case (opReduceSum):
    return "ReduceSum";
    break;
  case (opReduceSumSquare):
    return "ReduceSumSquare";
    break;
  case (opRelu):
    return "Relu";
    break;
  case (opRemainder):
    return "Remainder";
    break;
  case (opReshape):
    return "Reshape";
    break;
  case (opResize):
    return "Resize";
    break;
  case (opReverseSequence):
    return "ReverseSequence";
    break;
  case (opRoiAlign):
    return "RoiAlign";
    break;
  case (opRound):
    return "Round";
    break;
  case (opScan):
    return "Scan";
    break;
  case (opScatter):
    return "Scatter";
    break;
  case (opSelu):
    return "Selu";
    break;
  case (opShape):
    return "Shape";
    break;
  case (opShrink):
    return "Shrink";
    break;
  case (opSigmoid):
    return "Sigmoid";
    break;
  case (opSign):
    return "Sign";
    break;
  case (opSin):
    return "Sin";
    break;
  case (opSinh):
    return "Sinh";
    break;
  case (opSize):
    return "Size";
    break;
  case (opSlice):
    return "Slice";
    break;
  case (opSoftmax):
    return "Softmax";
    break;
  case (opSoftplus):
    return "Softplus";
    break;
  case (opSoftsign):
    return "Softsign";
    break;
  case (opSpaceToDepth):
    return "SpaceToDepth";
    break;
  case (opSplit):
    return "Split";
    break;
  case (opSqrt):
    return "Sqrt";
    break;
  case (opSqueeze):
    return "Squeeze";
    break;
  case (opStringNormalizer):
    return "StringNormalizer";
    break;
  case (opSub):
    return "Sub";
    break;
  case (opSum):
    return "Sum";
    break;
  case (opTan):
    return "Tan";
    break;
  case (opTanh):
    return "Tanh";
    break;
  case (opTfIdfVectorizer):
    return "TfIdfVectorizer";
    break;
  case (opThresholdedRelu):
    return "ThresholdedRelu";
    break;
  case (opTile):
    return "Tile";
    break;
  case (opTopK):
    return "TopK";
    break;
  case (opTranspose):
    return "Transpose";
    break;
  case (opTrueDiv):
    return "TrueDiv";
    break;
  case (opUnsqueeze):
    return "Unsqueeze";
    break;
  case (opUpsample):
    return "Upsample";
    break;
  case (opWhere):
    return "Where";
    break;
  case (opXor):
    return "Xor";
    break;
  case (opInvalid):
    return "Invalid";
    break;
  }
  return "Invalid";
}
} // namespace dnnc

#ifdef DNNC_OPTYPES_TEST
#include <iostream>

using namespace dnnc;

int main() {
  std::cout << getAttrNameStr(attr_transA);
  return 0;
}

#endif
