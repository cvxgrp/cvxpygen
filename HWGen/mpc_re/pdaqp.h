#ifndef PDAQP_H
#define PDAQP_H

typedef float c_float;
typedef float c_float_store;
typedef unsigned short c_int;
#define PDAQP_N_PARAMETER 6
#define PDAQP_N_SOLUTION 13

void pdaqp_evaluate(c_float* parameter, c_float* solution);
#endif // ifndef PDAQP_H
