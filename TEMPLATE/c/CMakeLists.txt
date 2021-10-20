
cmake_minimum_required (VERSION 3.5)

project (cvxpygen)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lm")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC

include_directories (include OSQP_code/include)

add_subdirectory (OSQP_code)

list (APPEND
      cpg_src
      ${CMAKE_CURRENT_SOURCE_DIR}/src/cpg_workspace.c
      ${CMAKE_CURRENT_SOURCE_DIR}/src/cpg_solve.c
      ${osqp_src}
)

add_executable (cpg_example src/cpg_example.c ${cpg_src})

add_library(cpg STATIC ${cpg_src})

set(cpg_src "${cpg_src}" PARENT_SCOPE)
