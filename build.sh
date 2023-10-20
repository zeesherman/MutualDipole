#!/bin/bash

# Load necessary Hyak modules
module load ssmc/miniconda/3.9
module load cuda/11.8.0
module load cmake/3.20.0
module load gcc/10.2.0

# Load hoomd by activating the python virtual environment in which HOOMD was built.
# Different file path needed for different users.
source /gscratch/zeelab/zsherm/hoomd/hoomd2.9.7-cuda11.8.0/hoomd-venv/bin/activate
rm -r build
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=/gscratch/zeelab/zsherm/hoomd/hoomd2.9.7-cuda11.8.0/hoomd-venv/lib/python3.9/site-packages/hoomd \
        -DCOPY_HEADERS=ON \
        -DCMAKE_CXX_FLAGS=-march=native \
        -DCMAKE_C_FLAGS=-march=native \
        -DENABLE_CUDA=ON \
        -DCMAKE_INCLUDE_PATH=/gscratch/zeelab/zsherm/amgx/amgx2.3.0-cuda11.8.0/AMGX/base/include \
        -DCMAKE_LIBRARY_PATH=/gscratch/zeelab/zsherm/amgx/amgx2.3.0-cuda11.8.0/AMGX/build 
make install