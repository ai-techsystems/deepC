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
#include "operators/baseOperator.h"
#include <string>

using namespace Eigen;

namespace dnnc {
template <typename To, typename Ti1, typename Ti2>

class LSTM : public baseOperator<To, Ti1, Ti2> {

  //
  // Type Constraints
  //
  // Ti1, To : tensor(float16), tensor(float), tensor(double)
  // Constrain input and output types to float tensors.
  //
  // Ti2 : tensor(int32)
  // Constrain seq_lens to integer tensor.
  //
  //

  //
  //  LSTM attributes
  //

protected:
  std::vector<float> activation_alpha;
  std::vector<float> activation_beta;
  // Optional scaling values used by some activation functions. The values are
  // consumed in the order of activation functions, for example (f, g, h) in
  // DLSTM. efault values are the same as of corresponding ONNX operators.

  std::vector<std::string> activations;
  // A list of 3 (or 6 if bidirectional) activation functions for input, output,
  // forget, cell, and hidden.

  float clip;
  // Cell clip threshold. Clipping bounds the elements of a tensor in the range
  // of [-threshold, +threshold] and is applied to the input of activations. No
  // clip if not specified.

  std::string direction;
  // Whether the RNN is forward, reverse, or bidirectional. Default is forward

  int hidden_size;
  // Number of neurons in the hidden layer

  int input_forget;
  // Couple input and forget gate if 1. Default is 0;

  int num_directions;
  // 2 - bidirectional, 1 - unidirectional

public:
  LSTM(std::string name = "opLSTM", std::vector<float> activation_alpha = {},
       std::vector<float> activation_beta = {},
       std::vector<std::string> activations = {}, float clip = 0,
       std::string direction = "forward", int hidden_size = 0,
       int input_forget = 0)
      : baseOperator<To, Ti1, Ti2>(opLSTM, name) {

    std::stringstream errMsg;
    std::vector<std::string> supported_activations = {"Relu", "Tanh",
                                                      "Sigmoid"};
    std::vector<std::string> valid_directions = {"forward", "reverse",
                                                 "bidirectional"};

    num_directions = (direction == "bidirectional") ? 2 : 1;

    // alpha and beta for activations
    if (activation_alpha.size() > 3 * num_directions) {
      errMsg << "Number of activation_alpha values (" << activation_alpha.size()
             << ") is more than " << 3 * num_directions << "for ";
      if (num_directions == 1) {
        errMsg << "unidirectional LSTM";
      } else {
        errMsg << "bidiretional LSTM";
      }
      errMsg << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
      return NULL_TENSOR<To>;
    }
    this->activation_alpha = activation_alpha;

    if (activation_beta.size() > 3 * num_directions) {
      errMsg << "Number of activation_beta values (" << activation_beta.size()
             << ") is more than " << 3 * num_directions << "for ";
      if (num_directions == 1) {
        errMsg << "unidirectional LSTM";
      } else {
        errMsg << "bidiretional LSTM";
      }
      SPDLOG_ERROR(errMsg.str().c_str());
      return NULL_TENSOR<To>;
    }
    this->activation_beta = activation_beta;

    // activation functions
    for (auto activation : activations) {
      if (std::find(activations.begin(), activations.end(), activation) ==
          supported_activations.end()) {
        errMsg << activation << " is not a supported activation funtion"
               << std::endl;
      }
      SPDLOG_ERROR(errMsg.str().c_str());
      return NULL_TENSOR<To>;
    }
    this->activations = activations;

    // clip
    this->clip = clip;

    // direction
    if (std::find(valid_directions.begin(), valid_directions.end(),
                  direction) == valid_directions.end()) {
      errMsg << direction << " is not a valid direction" << std::endl;
    }
    SPDLOG_ERROR(errMsg.str().c_str());
    return NULL_TENSOR<To>;
    this->direction = direction;

    // hidden_size?
    if (hidden_size < 0) {
      errMsg << hidden_size << "is not a valid value for hidden_size"
             << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
      return NULL_TENSOR<To>;
    }
    this->hidden_size = hidden_size;

    // input_forget?
    if ((input_forget < 0) || (input_forget > 1)) {
      errMsg << input_forget
             << "is not a valid value for input_forget ( must be 0 or 1)"
             << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
      return NULL_TENSOR<To>;
    }
    this->input_forget = input_forget;
  }

  bool getAttribute(OPATTR attrName, std::vector<float> &obj) override {
    if (attrName == attr_activation_alpha) {
      obj = activation_alpha;
      return true;
    } else if (attrName == attr_activation_beta) {
      obj = activation_beta;
      return true;
    }
    return false;
  }

  bool getAttribute(OPATTR attrName, std::vector<std::string> &obj) override {
    if (attrName == attr_activations) {
      obj = activations;
      return true;
    }
    return false;
  }

  bool getAttribute(OPATTR attrName, float &obj) override {
    if (attrName == attr_clip) {
      obj = clip;
      return true;
    }
    return false;
  }

  bool getAttribute(OPATTR attrName, std::string &obj) override {
    if (attrName == attr_direction) {
      obj = direction;
      return true;
    }
    return false;
  }

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_hidden_size) {
      obj = hidden_size;
      return true;
    } else if (attrName == attr_input_forget) {
      obj = input_forget;
      return true;
    }
    return false;
  }

  bool setAttribute(OPATTR attrName, std::vector<float> obj) override {
    if (attrName == attr_activation_alpha) {
      activation_alpha = obj;
      return true;
    } else if (attrName == attr_activation_beta) {
      activation_beta = obj;
      return true;
    }
    return false;
  }

  bool setAttribute(OPATTR attrName, std::vector<std::string> obj) override {
    if (attrName == attr_activations) {
      activations = obj;
      return true;
    }
    return false;
  }

  bool setAttribute(OPATTR attrName, float obj) override {
    if (attrName == attr_clip) {
      clip = obj;
      return true;
    }
    return false;
  }

  bool setAttribute(OPATTR attrName, std::string obj) override {
    if (attrName == attr_direction) {
      direction = obj;
      return true;
    }
    return false;
  }

  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_hidden_size) {
      hidden_size = obj;
      return true;
    } else if (attrName == attr_input_forget) {
      input_forget = obj;
      return true;
    }
    return false;
  }

  //
  // The compute funtion returns vector with 3 optional tensor outputs
  //
  // tensor<T> Y    [seq_length, num_directions, batch_size, hidden_size]
  // A tensor that concats all the intermediate output values of the hidden.
  //
  // tensor<T> Y_h  [num_directions, batch_size, hidden_size]
  // The last output value of the hidden
  //
  // tensor<T> Y_c  [num_directions, batch_size, hidden_size]
  // The last output value of the cell.
  //

  std::vector<tensor<To>> compute(
      //
      // Mandatory Inputs
      //

      tensor<Ti1> &X,
      // [seq_length, batch_size, input_size]
      // The input sequences 3-D tensor.

      tensor<Ti1> &W,
      // [num_directions, 4*hidden_size, input_size]
      // The weight tensor for the gates.
      // Concatenation of `W[iofc]` and `WB[iofc]` (if bidirectional) along
      // dimension 0

      tensor<Ti1> &R,
      // [num_directions, 4*hidden_size, hidden_size]
      // The recurrence weight tensor.
      // Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along
      // dimension 0.

      //
      // Optional Inputs
      //

      tensor<Ti1> &B = NULL_TENSOR<Ti1>,
      // [num_directions, 8*hidden_size]
      // The bias tensor for input gate.
      // Concatenation of `[Wb[iofc], Rb[iofc]]`, and `[WBb[iofc], RBb[iofc]]`
      // (if bidirectional) along dimension 0.

      tensor<Ti2> &sequence_lens = NULL_TENSOR<Ti2>,
      // [batch_size]
      // Lengths of the sequences in a batch.
      // If unspecified - assumed all sequences in the batch to have length
      // `seq_length`

      tensor<Ti1> &initial_h = NULL_TENSOR<Ti1>,
      // [num_directions, batch_size, hidden_size]
      // Initial value of the hidden. If unspecified - assumed to be 0

      tensor<Ti1> &initial_c = NULL_TENSOR<Ti1>,
      // [num_directions, batch_size, hidden_size]
      // Initial value of the cell. If unspecified - assumed to be 0

      tensor<Ti1> &P = NULL_TENSOR<Ti1>
      // [num_directions, batch_size, hidden_size]
      // The weight tensor for peepholes. If unspecified - assumed to be 0
  )

  {
    std::vector<tensor<To>> retVal;

    //
    // Process Attributes and Inputs
    //

    //
    // Compute
    //
    // Compute Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):
    // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    // Ct = ft (.) Ct-1 + it (.) ct
    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    // Ht = ot (.) h(Ct)

    return retVal;
  }

}; // class LSTM
} // namespace dnnc
