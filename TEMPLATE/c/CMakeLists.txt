
cmake_minimum_required (VERSION 3.5)

project (cvxpygen)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)

include_directories (include OSQP_code/include)

add_subdirectory (OSQP_code)

add_executable (cpg_example src/cpg_example.c src/cpg_workspace.c src/cpg_solve.c ${osqp_src})

add_library(cpg STATIC src/cpg_workspace.c src/cpg_solve.c ${osqp_src})
