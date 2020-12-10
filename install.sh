cd FaceKit/PCN/ 

# install library
sudo apt update
sudo apt install python3-opencv
sudo apt install libcaffe-cpu-dev
sudo apt install --no-install-recommends libboost-all-dev
sudo apt install libgflags-dev
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev python-dev libgflags-dev libatlas-base-dev libhdf5-serial-dev protobuf-compiler
sudo apt install libgflags-dev libgoogle-glog-dev liblmdb-dev

# build
make
sudo make install

cd ../../
