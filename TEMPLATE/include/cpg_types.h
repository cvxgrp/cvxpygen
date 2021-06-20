
#include "types.h"

#ifndef CPG_TYPES_H
# define CPG_TYPES_H

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

