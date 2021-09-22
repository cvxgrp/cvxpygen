
cmake_minimum_required (VERSION 3.5)

project (cvxpygen)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC

include_directories (include OSQP_code/include)

add_subdirectory (OSQP_code)

add_executable (cpg_example src/cpg_example.c src/cpg_workspace.c src/cpg_solve.c ${osqp_src})

add_library(cpg STATIC src/cpg_workspace.c src/cpg_solve.c ${osqp_src})
