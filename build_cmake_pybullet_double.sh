#!/bin/sh
rm CMakeCache.txt 
mkdir build_cmake 
cd build_cmake
cmake -DPYTHON_INCLUDE_DIR=/usr/include/python2.7
-DPYTHON_LIBRARY=$(python-config --prefix)/lib
-DUSE_SOFT_BODY_MULTI_BODY_DYNAMICS_WORLD=ON
-DBUILD_PYBULLET_MAC_USE_PYTHON_FRAMEWORK=OFF -DBUILD_PYBULLET=ON
-DBUILD_PYBULLET_NUMPY=OFF -DUSE_DOUBLE_PRECISION=ON -DCMAKE_BUILD_TYPE=Release .. 
#Build PyBullet using separate commando 
#CFLAGS="-DB3_NO_PYTHON_FRAMEWORK" python ../setup.py install 
make -j12 
cd examples 
cd pybullet 
ln -s pybullet.dylib pybullet.so
ln -s pybullet.dylib "`python -m site
--user-site`"\pybullet.so
