#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

extern "C" {
    #include "include/cpg_workspace.h"
    #include "include/cpg_solve.h"
    #include "OSQP_code/include/workspace.h"
}

namespace py = pybind11;

struct CPG_Info_cpp_t {
    double obj_val;
    int iter;
    char* status;
    double pri_res;
    double dua_res;
};

