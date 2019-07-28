sudo apt-get install autoconf automake libtool curl make g++ unzip
wget -P ../ https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-cpp-3.6.1.tar.gz
tar xvzf ../protobuf-cpp-3.6.1.tar.gz
cd ../protobuf-cpp-3.6.1
./configure
make
make check
sudo make install
sudo ldconfig