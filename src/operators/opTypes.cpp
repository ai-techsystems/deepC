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

std::string getAttrNameStr(OPATTR attr) {
  switch (attr) {
  case (attr_activation_alpha):
    return "activation_alpha";
  case (attr_activation_beta):
    return "activation_beta";
  case (attr_activations):
    return "activations";
  case (attr_alpha):
    return "alpha";
  case (attr_auto_pad):
    return "auto_pad";
  case (attr_axes):
    return "axes";
  case (attr_axis):
    return "axis";
  case (attr_batch_axis):
    return "batch_axis";
  case (attr_beta):
    return "beta";
  case (attr_bias):
    return "bias";
  case (attr_blocksize):
    return "blocksize";
  case (attr_body):
    return "body";
  case (attr_case_change_action):
    return "case_change_action";
  case (attr_ceil_mode):
    return "ceil_mode";
  case (attr_center_point_box):
    return "center_point_box";
  case (attr_clip):
    return "clip";
  case (attr_count_include_pad):
    return "count_include_pad";
  case (attr_detect_negative):
    return "detect_negative";
  case (attr_detect_positive):
    return "detect_positive";
  case (attr_dilations):
    return "dilations";
  case (attr_direction):
    return "direction";
  case (attr_dtype):
    return "dtype";
  case (attr_else_branch):
    return "else_branch";
  case (attr_epsilon):
    return "epsilon";
  case (attr_exclusive):
    return "exclusive";
  case (attr_fmod):
    return "fmod";
  case (attr_gamma):
    return "gamma";
  case (attr_group):
    return "group";
  case (attr_hidden_size):
    return "hidden_size";
  case (attr_high):
    return "high";
  case (attr_input_forget):
    return "input_forget";
  case (attr_is_case_sensitive):
    return "is_case_sensitive";
  case (attr_k):
    return "k";
  case (attr_keepdims):
    return "keepdims";
  case (attr_kernel_shape):
    return "kernel_shape";
  case (attr_lambd):
    return "lambd";
  case (attr_larges):
    return "larges";
  case (attr_linear_before_reset):
    return "linear_before_reset";
  case (attr_locale):
    return "locale";
  case (attr_low):
    return "low";
  case (attr_max_gram_length):
    return "max_gram_length";
  case (attr_max_skip_count):
    return "max_skip_count";
  case (attr_mean):
    return "mean";
  case (attr_min_gram_length):
    return "min_gram_length";
  case (attr_mode):
    return "mode";
  case (attr_momentum):
    return "momentum";
  case (attr_ngram_counts):
    return "ngram_counts";
  case (attr_ngram_indexes):
    return "ngram_indexes";
  case (attr_num_scan_inputs):
    return "num_scan_inputs";
  case (attr_output_height):
    return "output_height";
  case (attr_output_padding):
    return "output_padding";
  case (attr_output_shape):
    return "output_shape";
  case (attr_output_width):
    return "output_width";
  case (attr_p):
    return "p";
  case (attr_pads):
    return "pads";
  case (attr_perm):
    return "perm";
  case (attr_pool_int64s):
    return "pool_int64s";
  case (attr_pool_strings):
    return "pool_strings";
  case (attr_pooled_shape):
    return "pooled_shape";
  case (attr_ratio):
    return "ratio";
  case (attr_reverse):
    return "reverse";
  case (attr_sample_size):
    return "sample_size";
  case (attr_sampling_ratio):
    return "sampling_ratio";
  case (attr_scale):
    return "scale";
  case (attr_scan_input_axes):
    return "scan_input_axes";
  case (attr_scan_input_directions):
    return "scan_input_directions";
  case (attr_scan_output_axes):
    return "scan_output_axes";
  case (attr_scan_output_directions):
    return "scan_output_directions";
  case (attr_seed):
    return "seed";
  case (attr_shape):
    return "shape";
  case (attr_size):
    return "size";
  case (attr_sorted):
    return "sorted";
  case (attr_spatial_scale):
    return "spatial_scale";
  case (attr_split):
    return "split";
  case (attr_stopwords):
    return "stopwords";
  case (attr_storage_order):
    return "storage_order";
  case (attr_strides):
    return "strides";
  case (attr_then_branch):
    return "then_branch";
  case (attr_time_axis):
    return "time_axis";
  case (attr_to):
    return "to";
  case (attr_transA):
    return "transA";
  case (attr_transB):
    return "transB";
  case (attr_value):
    return "value";
  case (attr_weights):
    return "weights";
  case (attr_invalid):
    return "invalid";
  }
  return "invalid";
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
