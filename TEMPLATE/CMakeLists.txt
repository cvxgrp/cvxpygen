
cmake_minimum_required (VERSION 3.5)

project (cvxpygen)

include_directories (include)

add_executable (example src/example.c src/work.c src/update.c src/solve.c)