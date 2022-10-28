// Include the defined classes that are to be exported to python
#include "Dielectric2.h"

// Include pybind11
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

// specify the python module. Note that the name must explicitly match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
PYBIND11_MODULE(_Dielectric2, m)
    {
    #ifdef ENABLE_CUDA
	export_Dielectric2(m);
    #endif
    }