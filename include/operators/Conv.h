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
class Conv : public baseOperator<To, Ti1, Ti2> {
  //  Conv attributes
protected:
  std::string auto_pad;
  std::vector<int> dilations;
  int group;
  std::vector<int> kernel_shape;
  std::vector<int> pads;
  std::vector<int> strides;

public:
  Conv(std::string name = "opConv", std::string auto_pad = "NOTSET",
       std::vector<int> dilations = {}, int group = 1,
       std::vector<int> kernel_shape = {},
       // The shape of the convolution kernel. If not present, should be
       // inferred from input W.
       std::vector<int> pads = {}, std::vector<int> strides = {})
      : baseOperator<To, Ti1, Ti2>(opConv, name) {
    this->auto_pad = auto_pad;
    this->dilations = dilations;
    this->group = group;
    this->kernel_shape = kernel_shape;
    this->pads = pads;
    this->strides = strides;
  }

  bool getAttribute(OPATTR attrName, std::vector<int> &obj) override {
    if (attrName == attr_kernel_shape) {
      obj = kernel_shape;
      return true;
    } else if (attrName == attr_pads) {
      obj = pads;
      return true;
    } else if (attrName == attr_strides) {
      obj = strides;
      return true;
    } else if (attrName == attr_dilations) {
      obj = dilations;
      return true;
    }
    return false;
  }

  bool getAttribute(OPATTR attrName, int &obj) override {
    if (attrName == attr_group) {
      obj = group;
      return true;
    }
    return false;
  }

  bool getAttribute(OPATTR attrName, std::string &obj) override {
    if (attrName == attr_auto_pad) {
      obj = auto_pad;
      return true;
    }
    return false;
  }

  bool setAttribute(OPATTR attrName, std::string obj) override {
    if (attrName == attr_auto_pad) {
      auto_pad = obj;
      return true;
    }
    return false;
  }

  bool setAttribute(OPATTR attrName, std::vector<int> obj) override {
    if (attrName == attr_dilations) {
      dilations = obj;
      return true;
    }
    if (attrName == attr_kernel_shape) {
      kernel_shape = obj;
      return true;
    }
    if (attrName == attr_pads) {
      pads = obj;
      return true;
    }
    if (attrName == attr_strides) {
      strides = obj;
      return true;
    }
    return false;
  }

  bool setAttribute(OPATTR attrName, int obj) override {
    if (attrName == attr_group) {
      group = obj;
      return true;
    }
    return false;
  }

  tensor<To> compute(tensor<Ti1> &X, tensor<Ti1> &W,
                     tensor<Ti1> &B = NULL_TENSOR<Ti1>) {

    //
    // N - batch size
    // C = number of channels
    // H - Image height
    // W - Image width
    // M - number of feature maps
    //
    // X is Input data tensor from previous layer; has size (N x C x H x W),
    // In General, Input data tensor size could be (N x C x D1 x D2 ... x Dn).
    // For now supporting only for (N x C x H x W)

    // W is the weight tensor that will be used in the convolutions; has size
    // (M x C/group x kH x kW)

    // B it the optional 1D bias to be added to the convolution, has size of M

    // The output dimensions are functions of the kernel size, stride size, and
    // pad lengths.

    // Result shape is N x M x C x rH x rW
    // Result rH and rW formula R =(X-K+2P)/S + 1
    // Padding required on either side for same shape P = (X(S-1) - S + K)/2

    std::stringstream errMsg;

    //
    // basic initializations
    //

    // batch size
    size_t batchSize = X.shape()[0];

    // channels
    size_t numChannels = X.shape()[1];

    // data height and width
    size_t X_h = X.shape()[2];
    size_t X_w = X.shape()[3];

    // numFeatureMaps
    size_t numFeatureMaps = W.shape()[0];

    // result shape
    std::vector<size_t> resultShape;
    resultShape.push_back(batchSize);
    resultShape.push_back(numFeatureMaps);
    resultShape.push_back(numChannels);

    //
    // Process and check the arguments and inputs
    //

    // bias
    if (B.length() != numFeatureMaps) {
      errMsg << "Bias length (" << B.length()
             << "is different than number of feature maps (" << numFeatureMaps
             << ")" << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
      return NULL_TENSOR<To>;
    }

    // channels and groups
    if (W.shape()[1] != numChannels / group) {
      errMsg << "Weight tensor shape along second axis " << W.shape()[1]
             << "doesn't match " << numChannels / group
             << "(input channels/group)" << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
      return NULL_TENSOR<To>;
    }

    // stride
    if (strides.empty()) {
      // the stride defaults is 1 along each spatial axis.
      strides.push_back(1);
      strides.push_back(1);
    } else if (strides.size() != 2) {
      errMsg << "stride expected along 2 spatial axes, specified along"
             << strides.size() << "spatial axes" << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
      return NULL_TENSOR<To>;
    }

    // dilations
    if (dilations.empty()) {
      dilations = std::vector<int>(W.rank() - 2);
      // dilations defaults is 1 along each spatial axis
      for (size_t axis = 0; axis < W.rank() - 2; axis++) {
        dilations[axis] = 1;
      }
    } else if (dilations.size() != 2) {
      errMsg << "stride expected along 2 spatial axes, specified along"
             << dilations.size() << "spatial axes" << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
      return NULL_TENSOR<To>;
    }

    // Kernel
    std::vector<size_t> kernelShape(W.rank());
    kernelShape[0] = numFeatureMaps;
    kernelShape[1] = numChannels;
    for (size_t axis = 2; axis < W.rank(); axis++) {
      if (kernel_shape.empty()) {
        kernelShape[axis] = (W.shape()[axis]) * dilations[axis - 2];
      } else {
        kernelShape[axis] = kernel_shape[axis];
      }
    }

    tensor<Ti1> kernel(kernelShape);
    for (size_t featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
      for (size_t filterChannel = 0; filterChannel < numChannels;
           filterChannel++) {
        int channelIndx = filterChannel / group;
        for (size_t i = 0; i < kernelShape[2]; i++) {
          for (size_t j = 0; j < kernelShape[3]; j++) {
            if (((i + 1) % dilations[0] == 0) &&
                ((j + 1) % dilations[1] == 0)) {
              kernel(featureMap, filterChannel, i, j) =
                  W(featureMap, channelIndx, (((i + 1) / dilations[0]) - 1),
                    (((j + 1) / dilations[1]) - 1));
            } else {
              kernel(featureMap, filterChannel, i, j) = 0;
            }
          }
        }
      }
    }

    // auto_pad
    char padType = '\0';

    if (auto_pad == "VALID") {
      // no padding
      padType = 'N';
      for (size_t axis = 2; axis < X.rank(); axis++) {
        if (X.shape()[axis] <= kernelShape[axis]) {
          errMsg << "Kernel is too big for the given input and paddings"
                 << std::endl;
          SPDLOG_ERROR(errMsg.str().c_str());
          return NULL_TENSOR<To>;
        }
        resultShape.push_back(
            ((X.shape()[axis] - kernelShape[axis]) / strides[axis - 2]) + 1);
      }
      if (!pads.empty()) {
        errMsg << "auto_pad and pads attribute can't be used simultaneously"
               << std::endl;
        SPDLOG_ERROR(errMsg.str().c_str());
        return NULL_TENSOR<To>;
      }
    } else if (auto_pad == "SAME_UPPER") {
      padType = 'U';
      resultShape.push_back(X.shape()[2]);
      resultShape.push_back(X.shape()[3]);
      // add extra padding at the end to match the
      // output spatial size with the input
      if (!pads.empty()) {
        errMsg << "auto_pad and pads attribute can't be used simultaneously"
               << std::endl;
        SPDLOG_ERROR(errMsg.str().c_str());
        return NULL_TENSOR<To>;
      }
    } else if (auto_pad == "SAME_LOWER") {
      padType = 'L';
      resultShape.push_back(X.shape()[2]);
      resultShape.push_back(X.shape()[3]);
      // add extra padding at the beginning to match the
      // output spatial size with the input
      if (!pads.empty()) {
        errMsg << "auto_pad and pads attribute can't be used simultaneously"
               << std::endl;
        SPDLOG_ERROR(errMsg.str().c_str());
        return NULL_TENSOR<To>;
      }
    } else if (auto_pad == "NOTSET") {
      padType = 'P';
      if (pads.empty()) {
        errMsg << "explicit pads expected when auto_pad is \"NOTSET\""
               << std::endl;
        SPDLOG_ERROR(errMsg.str().c_str());
        return NULL_TENSOR<To>;
      }
      for (size_t axis = 2; axis < X.rank(); axis++) {
        if ((X.shape()[axis] + pads[axis] + pads[axis - 2]) <=
            kernelShape[axis]) {
          errMsg << "Kernel is too big for the given input and paddings"
                 << std::endl;
          SPDLOG_ERROR(errMsg.str().c_str());
          return NULL_TENSOR<To>;
        }
        resultShape.push_back(((X.shape()[axis] - kernelShape[axis] +
                                pads[axis] + pads[axis - 2]) /
                               strides[axis - 2]) +
                              1);
      }
    } else {
      errMsg << "auto_pad must be either \"NOTSET\", \"SAME_UPPER\", "
                "\"SAME_LOWER\" or \"VALID\""
             << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
      return NULL_TENSOR<To>;
    }

    // pads
    size_t padsSize = 2 * (X.rank() - 2);
    if (!pads.empty()) {
      if (pads.size() != (2 * (X.rank() - 2))) {
        errMsg << "pads expected format is [x1_begin, x2_begin...x1_end, "
                  "x2_end,...]"
               << "found " << pads.size() << " elements ( expected " << padsSize
               << std::endl;
        SPDLOG_ERROR(errMsg.str().c_str());
        return NULL_TENSOR<To>;
      }
      // above and below code is changed by Gunjan
      for (size_t i = 0; i < padsSize; i++) {
        if (pads[i] < 0) {
          errMsg << "pads value at index " << i << " is less than 0 ("
                 << pads[i] << ")" << std::endl;
          SPDLOG_ERROR(errMsg.str().c_str());
          return NULL_TENSOR<To>;
        }
      }
    } else {
      pads = std::vector<int>(padsSize);
      int total_padding_required;
      if (padType == 'N') {
        for (size_t i = 0; i < padsSize; i++) {
          pads[i] = 0;
        }
      } else if (padType == 'U') {
        for (size_t i = 0; i < X.rank() - 2; i++) {
          total_padding_required = X.shape()[i + 2] * (strides[i] - 1) -
                                   strides[i] + W.shape()[i + 2];
          pads[i] = pads[i + X.rank() - 2] = total_padding_required / 2;
          if ((total_padding_required % 2) != 0) {
            pads[i + X.rank() - 2] = pads[i + X.rank() - 2] + 1;
          }
        }

      } else if (padType == 'L') {
        for (size_t i = 0; i < X.rank() - 2; i++) {
          total_padding_required = X.shape()[i + 2] * (strides[i] - 1) -
                                   strides[i] + W.shape()[i + 2];
          pads[i] = pads[i + X.rank() - 2] = total_padding_required / 2;
          if ((total_padding_required % 2) != 0) {
            pads[i] = pads[i] + 1;
          }
        }
      }
    }

    // work out the result
    tensor<To> result(resultShape);
    tensor<Ti1> paddedInput({X_h + pads[0] + pads[2], X_w + pads[1] + pads[3]});
    tensor<Ti1> convImage({resultShape[3], resultShape[4]});
    tensor<Ti1> filter({kernelShape[2], kernelShape[3]});
    std::vector<size_t> __pads;

    for (size_t i = 0; i < pads.size(); i++) {
      __pads.push_back((size_t)pads[i]); // need to do this for type conversion
    }

    for (size_t batchIndx = 0; batchIndx < batchSize; batchIndx++) {
      for (size_t channelIndx = 0; channelIndx < numChannels; channelIndx++) {

        // padded input image
        for (size_t hIndx = 0; hIndx < X_h + __pads[0] + __pads[2]; hIndx++) {
          for (size_t wIndx = 0; wIndx < X_w + __pads[1] + __pads[3]; wIndx++) {
            if (hIndx < __pads[0] || hIndx >= (X_h + __pads[0]) ||
                wIndx < __pads[1] || wIndx >= (X_w + __pads[1])) {
              paddedInput(hIndx, wIndx) = 0;
            } else {
              paddedInput(hIndx, wIndx) = X(
                  batchIndx, channelIndx, hIndx - __pads[0], wIndx - __pads[1]);
            }
          }
        }

        for (size_t featureMapIndx = 0; featureMapIndx < numFeatureMaps;
             featureMapIndx++) {

          // convolve
          for (size_t hIndx = 0; hIndx < resultShape[3]; hIndx = hIndx + 1) {
            for (size_t wIndx = 0; wIndx < resultShape[4]; wIndx = wIndx + 1) {
              result(batchIndx, featureMapIndx, channelIndx, hIndx, wIndx) = 0;
              if (B != NULL_TENSOR<Ti1>) {
                result(batchIndx, featureMapIndx, channelIndx, hIndx, wIndx) =
                    B(featureMapIndx);
              }
              for (size_t i = 0; i < kernelShape[2]; i++) {
                for (size_t j = 0; j < kernelShape[3]; j++) {
                  result(batchIndx, featureMapIndx, channelIndx, hIndx,
                         wIndx) += kernel(featureMapIndx, channelIndx, i, j) *
                                   paddedInput(hIndx * (size_t)strides[1] + i,
                                               wIndx * (size_t)strides[1] + j);
                }
              }
            }
          }
          // end convolve
        }
      }
    }
    return result;
  }

}; // template class
} // namespace dnnc
