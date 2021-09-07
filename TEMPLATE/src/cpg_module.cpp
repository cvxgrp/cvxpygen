#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

extern "C" {
    #include "cpg_workspace.h"
    #include "cpg_solve.h"
}

namespace py = pybind11;

