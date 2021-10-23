
cmake_minimum_required (VERSION 3.5)

project (cvxpygen)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lm")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC

include_directories (include solver_code/include)

add_subdirectory (solver_code)

list (APPEND
      cpg_head
      ${CMAKE_CURRENT_SOURCE_DIR}/include/cpg_workspace.h
      ${CMAKE_CURRENT_SOURCE_DIR}/include/cpg_solve.h
      ${solver_head}
)

list (APPEND
      cpg_src
      ${CMAKE_CURRENT_SOURCE_DIR}/src/cpg_workspace.c
      ${CMAKE_CURRENT_SOURCE_DIR}/src/cpg_solve.c
      ${solver_src}
)

add_executable (cpg_example ${cpg_head} src/cpg_example.c ${cpg_src})

add_library(cpg STATIC ${cpg_head} ${cpg_src})

set(cpg_head "${cpg_head}" PARENT_SCOPE)
set(cpg_src "${cpg_src}" PARENT_SCOPE)
