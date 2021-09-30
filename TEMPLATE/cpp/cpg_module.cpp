#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>

extern "C" {
    #include "include/cpg_workspace.h"
    #include "include/cpg_solve.h"
    #include "OSQP_code/include/workspace.h"
}

namespace py = pybind11;

