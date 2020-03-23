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

#define _USE_EIGEN 1

#define TENSOR_DIMENSIONS_EQUAL(t1, t2)                                        \
  { t1.size() == t2.size() }

/*<! python interface tensor print limit */
#define DNNC_TENSOR_MAX_EL 30

#if defined(WIN32) || defined(_WIN32)
#define FS_PATH_SEPARATOR "\\"
#else
#define FS_PATH_SEPARATOR "/"
#endif
namespace dnnc {
typedef size_t INDEX;
typedef size_t DIMENSION;
} // namespace dnnc

#if defined(__clang__)
#if __has_feature(cxx_rtti)
#define RTTI_ENABLED
#endif
#elif defined(__GNUG__)
#if defined(__GXX_RTTI)
#define RTTI_ENABLED
#endif
#elif defined(_MSC_VER)
#if defined(_CPPRTTI)
#define RTTI_ENABLED
#endif
#endif

#if defined(ARDUINO)
#define SPDLOG_ERROR(msg)                                                      \
  {                                                                            \
    Serial.print("Warning: ");                                                 \
    Serial.print(__FILE__);                                                    \
    Serial.print(":");                                                         \
    Serial.print(__LINE__);                                                    \
    Serial.print(msg);                                                         \
    Serial.println();                                                          \
    Serial.flush();                                                            \
  }
#define SPDLOG_WARN(msg)                                                       \
  {                                                                            \
    Serial.print("Error: ");                                                   \
    Serial.print(__FILE__);                                                    \
    Serial.print(":");                                                         \
    Serial.print(__LINE__);                                                    \
    Serial.print(msg);                                                         \
    Serial.println();                                                          \
    Serial.flush();                                                            \
  }
#else
#define SPDLOG_WARN(msg) std::cout << "Warn: " << msg << std::endl;
#define SPDLOG_ERROR(msg) std::cout << "Error: " << msg << std::endl;
#endif
