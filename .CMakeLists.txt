# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(dnnc C CXX)

# find the protobuf compiler and libraries
find_package(Protobuf REQUIRED)

# check if protobuf was found
if(PROTOBUF_FOUND)
    message ("protobuf found")
else()
    message (FATAL_ERROR "Cannot find Protobuf")
endif()

# Generate the .h and .cxx files
set(DNNC_ROOT ${PROJECT_SOURCE_DIR})

FILE(GLOB onnx_PROTO_FILES onnx/*.proto)
FILE(GLOB DNNC_SRCS src/*.cc)

set(PROTOBUF_IMPORT_DIRS "${PROJECT_SOURCE_DIR}/onnx")

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${onnx_PROTO_FILES})

# DEBUG, Print path to generated files
message ("PROTOBUF_IMPORT_DIRS = ${PROTOBUF_IMPORT_DIRS}")
message ("onnx_PROTO_FILES = ${onnx_PROTO_FILES}")
message ("PROTO_SRCS = ${PROTO_SRCS}")
message ("PROTO_HDRS = ${PROTO_HDRS}")

# Add an executable
add_executable(dnnc
    ${DNNC_SRCS}
    ${PROTO_SRCS}
    ${PROTO_HDRS})

target_include_directories(dnnc
    PUBLIC
    ${PROTOBUF_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# link the exe against the libraries
target_link_libraries(dnnc
    PUBLIC
    ${PROTOBUF_LIBRARIES}
)
