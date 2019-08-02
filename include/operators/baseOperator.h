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
#include <tensor.h>

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

  // operator attributes
  struct opAttributes {
    // it'll be a map of key-value pair, that
    // operators will customize for their needs.
  };

	template <typename T>
  class baseOperator 
  {
    protected:
      OPCODE             _op;
      opAttributes*      _m_attrs ;
      std::vector<std::shared_ptr<tensor<T> > > _m_inputs ;
      std::vector<std::shared_ptr<tensor<T> > > _m_outputs ;

    public:
      baseOperator(OPCODE op, opAttributes* attr=0x0) : _op(op), _m_attrs(attr) 
    {}
      void addInput(std::shared_ptr<tensor<T> > in) 
      {
        _m_inputs.push_back(in);
      }
      void addOutput(std::shared_ptr<tensor<T> > out) 
      {
        _m_inputs.push_back(out);
      }
  };
}
