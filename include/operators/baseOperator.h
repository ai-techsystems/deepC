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
#include "core/broadcast.h"
#include "operators/macros.h"
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

// we're forced to include tensor.h here, because of limitation on
// template instantiations to generate complete definition of the
// operator. This breaks principle of modularity along with my heart. :-/
#include "core/tensor.h"

namespace dnnc {

enum OPCODE {
  opAbs = 1,
  opAcos,
  opAcosh,
  opAdd,
  opAnd,
  opArgMax,
  opArgMin,
  opAsin,
  opAsinh,
  opAtan,
  opAtanh,
  opAveragePool,
  opBatchNormalization,
  opBitShift,
  opCast,
  opCeil,
  opClip,
  opCompress,
  opConcat,
  opConstant,
  opConstantOfShape,
  opConv,
  opConvInteger,
  opConvTranspose,
  opCos,
  opCosh,
  opCumSum,
  opDepthToSpace,
  opDequantizeLinear,
  opDiv,
  opDropout,
  opElu,
  opEqual,
  opErf,
  opExp,
  opExpand,
  opEyeLike,
  opFlatten,
  opFloor,
  opGRU,
  opGather,
  opGemm,
  opGlobalAveragePool,
  opGlobalLpPool,
  opGlobalMaxPool,
  opGreater,
  opHardSigmoid,
  opHardmax,
  opIdentity,
  opIf,
  opInstanceNormalization,
  opIsInf,
  opIsNaN,
  opLRN,
  opLSTM,
  opLeakyRelu,
  opLess,
  opLog,
  opLogSoftmax,
  opLoop,
  opLpNormalization,
  opLpPool,
  opMatMul,
  opMatMulInteger,
  opMax,
  opMaxPool,
  opMaxRoiPool,
  opMaxUnpool,
  opMean,
  opMin,
  opMod,
  opMul,
  opMultinomial,
  opNeg,
  opNonMaxSuppression,
  opNonZero,
  opNot,
  opOneHot,
  opOr,
  opPRelu,
  opPad,
  opPow,
  opQLinearConv,
  opQLinearMatMul,
  opQuantizeLinear,
  opRNN,
  opRandomNormal,
  opRandomNormalLike,
  opRandomUniform,
  opRandomUniformLike,
  opReciprocal,
  opReduceL1,
  opReduceL2,
  opReduceLogSum,
  opReduceLogSumExp,
  opReduceMax,
  opReduceMean,
  opReduceMin,
  opReduceProd,
  opReduceSum,
  opReduceSumSquare,
  opRelu,
  opReshape,
  opResize,
  opReverseSequence,
  opRoiAlign,
  opRound,
  opScan,
  opScatter,
  opSelu,
  opShape,
  opShrink,
  opSigmoid,
  opSign,
  opSin,
  opSinh,
  opSize,
  opSlice,
  opSoftmax,
  opSoftplus,
  opSoftsign,
  opSpaceToDepth,
  opSplit,
  opSqrt,
  opSqueeze,
  opStringNormalizer,
  opSub,
  opSum,
  opTan,
  opTanh,
  opTfIdfVectorizer,
  opThresholdedRelu,
  opTile,
  opTopK,
  opTranspose,
  opUnsqueeze,
  opUpsample,
  opWhere,
  opXor
};

enum OPATTR {
  attr_activation_alpha = 1,
  attr_activation_beta,
  attr_activations,
  attr_alpha,
  attr_auto_pad,
  attr_axes,
  attr_axis,
  attr_batch_axis,
  attr_beta,
  attr_bias,
  attr_blocksize,
  attr_body,
  attr_case_change_action,
  attr_ceil_mode,
  attr_center_point_box,
  attr_clip,
  attr_count_include_pad,
  attr_detect_negative,
  attr_detect_positive,
  attr_dilations,
  attr_direction,
  attr_dtype,
  attr_else_branch,
  attr_epsilon,
  attr_exclusive,
  attr_fmod,
  attr_gamma,
  attr_group,
  attr_hidden_size,
  attr_high,
  attr_input_forget,
  attr_is_case_sensitive,
  attr_k,
  attr_keepdims,
  attr_kernel_shape,
  attr_lambd,
  attr_larges,
  attr_linear_before_reset,
  attr_locale,
  attr_low,
  attr_max_gram_length,
  attr_max_skip_count,
  attr_mean,
  attr_min_gram_length,
  attr_mode,
  attr_momentum,
  attr_ngram_counts,
  attr_ngram_indexes,
  attr_num_scan_inputs,
  attr_output_height,
  attr_output_padding,
  attr_output_shape,
  attr_output_width,
  attr_p,
  attr_pads,
  attr_perm,
  attr_pool_int64s,
  attr_pool_strings,
  attr_pooled_shape,
  attr_ratio,
  attr_reverse,
  attr_sample_size,
  attr_sampling_ratio,
  attr_scale,
  attr_scan_input_axes,
  attr_scan_input_directions,
  attr_scan_output_axes,
  attr_scan_output_directions,
  attr_seed,
  attr_shape,
  attr_size,
  attr_sorted,
  attr_spatial_scale,
  attr_split,
  attr_stopwords,
  attr_storage_order,
  attr_strides,
  attr_then_branch,
  attr_time_axis,
  attr_to,
  attr_transA,
  attr_transB,
  attr_value,
  attr_weights
};

template <typename T> class baseOperator {
protected:
  OPCODE _op;
  std::string _name;

  T *tensorMem(tensor<T> &t) { return t._mem_layout; }

public:
  baseOperator(OPCODE op, std::string name = "") : _op(op), _name(name) {}

  /*!< return name of the operator */
  inline std::string name() { return _name; }
  /*!< return OPCODE of the operator */
  inline OPCODE symbol() { return _op; }

  template <typename attrType>
  bool getAttribute(OPATTR attrName, attrType &obj);

  /*!< Constrain input and output types for onnx.*/
  template <typename... Types> bool type_check() {
    std::vector<std::type_index> allowed_types;
    allowed_types.insert(allowed_types.end(), {typeid(Types)...});
    bool checker = false;
    for (size_t i = 0; i < allowed_types.size(); i++) {
      checker = (allowed_types[i] == std::type_index(typeid(T)));
      if (checker)
        break;
    }
    return checker;
  }
  /*!<
   \return Returns true if T is one of the types specified
   */
  virtual void compute(void);
  virtual tensor<T> compute(tensor<T> in1);
  virtual tensor<T> compute(tensor<T> &in1);
  virtual tensor<T> compute(tensor<T> in1, tensor<T> in2);
  virtual tensor<T> compute(tensor<T> &in1, tensor<T> &in2);
};

template <typename T> struct opCmp {
  bool operator()(const baseOperator<T> &lhs, const baseOperator<T> &rhs) {
    return lhs.symbol() == rhs.symbol() ? lhs.name() < rhs.name()
                                        : lhs.symbol() < rhs.symbol();
  }
};

} // namespace dnnc
