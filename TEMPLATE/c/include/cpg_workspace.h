
#include "types.h"

#ifndef CPG_TYPES_H
# define CPG_TYPES_H

typedef struct {
    int         P;              ///< bool, if OSQP parameter P outdated
    int         q;              ///< bool, if OSQP parameter q outdated
    int         d;              ///< bool, if OSQP parameter d outdated
    int         A;              ///< bool, if OSQP parameter A outdated
    int         l;              ///< bool, if OSQP parameter l outdated
    int         u;              ///< bool, if OSQP parameter u outdated
} OSQP_Outdated_t;

// Struct containing flags for outdated OSQP parameters
extern OSQP_Outdated_t OSQP_Outdated;

typedef struct {
    csc         *P;              ///< OSQP parameter P
    c_float     *q;              ///< OSQP parameter q
    c_float     *d;              ///< OSQP parameter d
    csc         *A;              ///< OSQP parameter A
    c_float     *l;              ///< OSQP parameter l
    c_float     *u;              ///< OSQP parameter u
} OSQP_Params_t;

