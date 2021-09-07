
cmake_minimum_required (VERSION 3.5)

project (cvxpygen)

include_directories (include OSQP_code/include)

add_subdirectory (OSQP_code)

add_executable (cpg_example src/cpg_example.c src/cpg_workspace.c src/cpg_solve.c ${osqp_src})

add_subdirectory(pybind11)

pybind11_add_module(cpg_module src/cpg_module.cpp src/cpg_workspace.c src/cpg_solve.c ${osqp_src})