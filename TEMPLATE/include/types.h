
#ifndef CPG_TYPES_H
# define CPG_TYPES_H

typedef int c_int;
typedef float c_float;

typedef struct {
    c_int    nzmax; ///< maximum number of entries
    c_int    m;     ///< number of rows
    c_int    n;     ///< number of columns
    c_int   *p;     ///< column pointers (size n+1); col indices (size nzmax) start from 0 when using triplet format (direct KKT matrix formation)
    c_int   *i;     ///< row indices, size nzmax starting from 0
    c_float *x;     ///< numerical values, size nzmax
    c_int    nz;    ///< number of entries in triplet matrix, -1 for csc
} csc;

typedef struct {
    csc         *P;              ///< OSQP parameter P
    c_float     *q;              ///< OSQP parameter q
    c_float     *d;              ///< OSQP parameter d
    csc         *A;              ///< OSQP parameter A
    c_float     *l;              ///< OSQP parameter l
    c_float     *u;              ///< OSQP parameter u
    c_float     *P_decomposed;   ///< decomposition of OSQP data vector
    c_float     *q_decomposed;   ///< decomposition of OSQP data vector
    c_float     *d_decomposed;   ///< decomposition of OSQP data vector
    c_float     *A_decomposed;   ///< decomposition of OSQP data vector
    c_float     *l_decomposed;   ///< decomposition of OSQP data vector
    c_float     *u_decomposed;   ///< decomposition of OSQP data vector
} OSQP_Workspace_t;

