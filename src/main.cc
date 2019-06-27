#include <iostream>
#include <fstream>
#include <string>
#include "onnx.pb.h"

int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 2) {
    std::cerr << "Usage:  " << argv[0] << " onnx_model_file" << "\n";
    return -1;
  }
  std::cout << "DNNC is under development. Check back in Aug 2019 for full release.\n";

  return 0;
}
