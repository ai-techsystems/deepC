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
#include "operators/baseOperator.h"
#include <string>

using namespace Eigen;

namespace dnnc {
template <typename To, typename Ti, typename Tind>
class SetSlice : public baseOperator<To, Ti, Tind> {
  //  SetSlice attributes
public:
  SetSlice(std::string name = "opSetSlice")
      : baseOperator<To, Ti, Tind>(opSetSlice, name) {}

  // bool getAttribute<int>(OPATTR attrName, int& obj) ;

  void compute(tensor<To> output,  // N-D Tensor of data to set data to
               tensor<Ti> input,   // N-D Tensor of data to extract from
               tensor<Tind> start, // 1-D tensor of starting indices of
                                   // corresponding axis in `axes`
               tensor<Tind> end, // 1-D tensor of ending indices (exclusive) of
                                 // corresponding axis in `axes`
               tensor<Tind> axes = NULL_TENSOR<Tind>,
               // 1-D tensor of axes that `starts` and `ends` apply to.
               // Negative value means counting dimensions from the back.
               tensor<Tind> steps = NULL_TENSOR<Tind>)

  // 1-D tensor of slice step of corresponding
  // axis in `axes`. Default to 1.
  {
    // Process and check the arguments

    std::stringstream errMsg;

    DIMENSION num_axes = start.shape()[0];
    Tind rank = output.rank();

    if (start.rank() != 1) {
      errMsg << "start tensor is " << start.rank()
             << "dimensional (should be 1 dimensional)" << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
    }

    if (end.rank() != 1) {
      errMsg << "end tensor is " << end.rank()
             << "dimensional (should be 1 dimensional)" << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
    }

    if (start.shape() != end.shape()) {
      errMsg << "start and end tensor sizes don't match (";
      errMsg << "start tensor size = " << start.shape()[0] << ", ";
      errMsg << "end tensor size = " << end.shape()[0] << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
    }

    if (axes == NULL_TENSOR<int>) {
      std::vector<DIMENSION> shape{num_axes};
      tensor<int> default_axis(shape);
      axes = default_axis;
      for (size_t i = 0; i < num_axes; i++) {
        axes(i) = i;
      }
    }

    if (steps == NULL_TENSOR<DIMENSION>) {
      std::vector<DIMENSION> shape{num_axes};
      tensor<Tind> default_steps(shape);
      steps = default_steps;
      for (size_t i = 0; i < num_axes; i++) {
        steps(i) = 1;
      }
    }

    if (axes.rank() != 1) {
      errMsg << "axes tensor is " << axes.rank()
             << "dimensional (should be 1 dimensional)" << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
    }

    if (steps.rank() != 1) {
      errMsg << "steps tensor is " << steps.rank()
             << "dimensional (should be 1 dimensional)" << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
    }

    if (start.shape() != axes.shape()) {
      errMsg << "start and axes tensor sizes don't match (";
      errMsg << "start tensor size = " << start.shape()[0] << ", ";
      errMsg << "axes tensor size = " << axes.shape()[0] << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
    }

    if (start.shape() != steps.shape()) {
      errMsg << "start and axes tensor sizes don't match (";
      errMsg << "start tensor size = " << start.shape()[0] << ", ";
      errMsg << "steps tensor size = " << steps.shape()[0] << std::endl;
      SPDLOG_ERROR(errMsg.str().c_str());
    }

    for (size_t i = 0; i < num_axes; i++) {

      // change values from negative to positive
      if (start(i) < 0) {
        start(i) += output.shape()[i];
      }
      if (end(i) < 0) {
        // when step is negative and end is -1, store -1
        // this is required by python_slice for negative steps
        if ((steps(i) < 0) && (end(i) == -1)) {
          end(i) = -1;
        } else {
          end(i) += output.shape()[i];
        }
      }

      // Numpy like checks and counter measures for corner cases
      // step cannot be zero
      if (steps(i) == 0) {
        errMsg << "slice step cannot be zero" << std::endl;
        SPDLOG_ERROR(errMsg.str().c_str());
      }
      // if step is positive
      else if (steps(i) > 0) {
        // return NULL tensor if start is greater equal to
        // shape[axis] or start is greater equal to end
        if ((Tind)start(i) >= (Tind)output.shape()[i] ||
            (end(i) - 1 < start(i))) {
          return;
        }
        // if end is greater than shape[axis], limit end to shape[axis]
        if ((Tind)end(i) > (Tind)(output.shape()[i])) {
          end(i) = output.shape()[i];
        }
      }
      // if step is negative
      else if (steps(i) < 0) {
        // if start is greater equal to shape[axis], limit start to
        // shape[axis]-1
        if ((Tind)start(i) >= (Tind)output.shape()[i]) {
          start(i) = output.shape()[i] - 1;
        }
        // return NULL tensor if end is greater equal to
        // shape[axis] or end is greater equal to start
        if ((Tind)end(i) >= (Tind)(output.shape()[i]) ||
            (start(i) - 1 < end(i))) {
          return;
        }
      }

      // ** Numpy doen't raise error for the below conditions,
      // it smartly avoids them

      // start
      // if (start(i) > output.shape()[i]) {
      //   errMsg << "start value (" << start(i) << ") along axis (" << i
      //          << ") is beyond the size (" << output.shape()[i]
      //          << ") of input tensor along the axis" << std::endl;
      //   SPDLOG_ERROR(errMsg.str().c_str());
      // }

      // end
      // if (end(i) > (output.shape()[i])) {
      //   errMsg << "end value (" << end(i) << ") along axis (" << i
      //          << ") is beyond the size (" << output.shape()[i]
      //          << ") of input tensor along the axis" << std::endl;
      //   SPDLOG_ERROR(errMsg.str().c_str());
      // }

      // comparing start and end when step is positive
      // else if ((steps(i) > 0) && (end(i) - 1 < start(i))) {
      //   errMsg << "end value (" << end(i) - 1 << ") along axis (" << i
      //          << ") is smaller than the start value (" << start(i)
      //          << ") along the axis while step is positive" << std::endl;
      //   SPDLOG_ERROR(errMsg.str().c_str());
      // }

      // comparing start and end when step is negative
      // else if ((steps(i) < 0) && (start(i) - 1 < end(i))) {
      // errMsg << "start value (" << start(i) - 1 << ") along axis (" << i
      //        << ") is smaller than the end value (" << end(i)
      //        << ") along the axis while step is negative" << std::endl;
      // SPDLOG_ERROR(errMsg.str().c_str());
      // }

      // axes
      if (axes(i) < 0) {
        if ((axes(i) + rank) < 0) {
          errMsg << "axes value (" << axes(i) << ") along axis (" << i
                 << ") is beyond the input tensor dimension" << std::endl;
          SPDLOG_ERROR(errMsg.str().c_str());
        }
        axes(i) = rank + axes(i);
      }
      if (axes(i) > rank - 1) {
        errMsg << "axes value (" << axes(i) << ") along axis (" << i
               << ") is large than the number of dimensions of input tensor"
               << std::endl;
        SPDLOG_ERROR(errMsg.str().c_str());
      }

      for (size_t j = i + 1; j < num_axes; j++) {
        if (axes(i) == axes(j)) {
          errMsg << "repeated axis value (" << axes(i) << ") at indices " << i
                 << " and " << j << " of axes input" << std::endl;
          SPDLOG_ERROR(errMsg.str().c_str());
        }
      }

      // steps
    }

    // Determine the shape of the result tensor

    std::vector<size_t> resultShape(rank);
    std::vector<Tind> start_index(rank);
    std::vector<Tind> end_index(rank);
    std::vector<Tind> step(rank);

    for (int axis = 0; axis < rank; axis++) {
      // determine slicing along the axis-th dimension
      for (size_t i = 0; i < num_axes; i++) {
        if (axes(i) == axis) {
          if (steps[i] > 0) {
            start_index[axis] = start(i);
            end_index[axis] = end(i) - 1;
            step[axis] = steps[i];
          } else {
            // Changed by Gunjan, marked to find it later if doesn't work
            start_index[axis] = start(i);
            end_index[axis] = end(i) + 1;
            step[axis] = steps[i];
            /*
            int tmp_start = start(i);
            end_index[axis] = start(i);
            while (tmp_start > end(i)) {
              start_index[axis] = start;
              tmp_start = tmp_start + steps[i];
            }
            step[axis] = -steps[i];
            */
          }
          break;
        } else {
          start_index[axis] = 0;
          end_index[axis] = output.shape()[axis] - 1;
          step[axis] = 1;
        }
      }
      resultShape[axis] =
          (end_index[axis] - start_index[axis]) / step[axis] + 1;
    }

    // Try broadcasting now

    if (resultShape != input.shape()) {
      input = broadcast(input, resultShape);
      if (input.isnull()) {
        errMsg << "could not broadcast input array" << std::endl;
        SPDLOG_ERROR(errMsg.str().c_str());
      }
    }

    // SetSlice now

    if (rank == 1) {
      Tind i0 = 0;
      for (Tind _i0 = start_index[0];
           (step[0] > 0) ? (_i0 <= end_index[0]) : (_i0 >= end_index[0]);
           _i0 += step[0]) {
        output(_i0) = static_cast<To>(input(i0++));
      }
    } else if (rank == 2) {
      Tind i0 = 0;
      for (Tind _i0 = start_index[0];
           (step[0] > 0) ? (_i0 <= end_index[0]) : (_i0 >= end_index[0]);
           _i0 += step[0]) {
        Tind i1 = 0;
        for (Tind _i1 = start_index[1];
             (step[1] > 0) ? (_i1 <= end_index[1]) : (_i1 >= end_index[1]);
             _i1 += step[1]) {
          // std::cout << _i0 << " , " << _i1 << " : " << output(_i0,_i1) <<
          // std::endl;  // for testing purposes
          output(_i0, _i1) = static_cast<To>(input(i0, i1++));
        }
        i0++;
      }
    } else if (rank == 3) {
      Tind i0 = 0;
      for (Tind _i0 = start_index[0];
           (step[0] > 0) ? (_i0 <= end_index[0]) : (_i0 >= end_index[0]);
           _i0 += step[0]) {
        Tind i1 = 0;
        for (Tind _i1 = start_index[1];
             (step[1] > 0) ? (_i1 <= end_index[1]) : (_i1 >= end_index[1]);
             _i1 += step[1]) {
          Tind i2 = 0;
          for (Tind _i2 = start_index[2];
               (step[2] > 0) ? (_i2 <= end_index[2]) : (_i2 >= end_index[2]);
               _i2 += step[2]) {
            output(_i0, _i1, _i2) = static_cast<To>(input(i0, i1, i2++));
          }
          i1++;
        }
        i0++;
      }
    } else if (rank == 4) {
      Tind i0 = 0;
      for (Tind _i0 = start_index[0];
           (step[0] > 0) ? (_i0 <= end_index[0]) : (_i0 >= end_index[0]);
           _i0 += step[0]) {
        Tind i1 = 0;
        for (Tind _i1 = start_index[1];
             (step[1] > 0) ? (_i1 <= end_index[1]) : (_i1 >= end_index[1]);
             _i1 += step[1]) {
          Tind i2 = 0;
          for (Tind _i2 = start_index[2];
               (step[2] > 0) ? (_i2 <= end_index[2]) : (_i2 >= end_index[2]);
               _i2 += step[2]) {
            Tind i3 = 0;
            for (Tind _i3 = start_index[3];
                 (step[3] > 0) ? (_i3 <= end_index[3]) : (_i3 >= end_index[3]);
                 _i3 += step[3]) {
              output(_i0, _i1, _i2, _i3) =
                  static_cast<To>(input(i0, i1, i2, i3++));
            }
            i2++;
          }
          i1++;
        }
        i0++;
      }
    } else {
      SPDLOG_ERROR("Not supported");
    }
    return;
  }
};
} // namespace dnnc
