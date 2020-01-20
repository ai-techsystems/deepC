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

// TEMPORARY_LOGGER_IMPLEMENTATION
/*
    Logger needs it's own class
*/

// colour codes to print on terminal
#define log_red "\033[0;31m"
#define log_green "\033[1;32m"
#define log_yellow "\033[1;33m"
#define log_cyan "\033[0;36m"
#define log_magenta "\033[0;35m"
#define log_reset "\033[0m"

// to only get file name instead of full file path
#define __FILENAME__ (__builtin_strrchr(__FILE__, '/') ?                        \
        __builtin_strrchr(__FILE__, '/') + 1 : __FILE__)

#define LOG_ERROR(message)              \
    (std::cout << "[" << log_red << "ERROR"  << log_reset << "] | " << message   \
        << " | [" << log_red << __FILENAME__ << log_reset << "] | [" <<          \
        log_red << __LINE__ << log_reset << "]" << std::endl)

#define LOG_WARN(message)              \
    (std::cout << "[" << log_yellow << "WARNING"  << log_reset << "] | " << message   \
        << " | [" << log_yellow << __FILENAME__ << log_reset << "] | [" <<          \
        log_yellow << __LINE__ << log_reset << "]" << std::endl)

#define LOG_INFO(message)              \
    (std::cout << "[" << log_green << "INFO"  << log_reset << "] | " << message   \
        << " | [" << log_green << __FILENAME__ << log_reset << "] | [" <<          \
        log_green << __LINE__ << log_reset << "]" << std::endl)