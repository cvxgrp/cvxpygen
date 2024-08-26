
#include "cpg_workspace.h"

#ifndef CPG_OSQP_GRAD_TYPES_H
# define CPG_OSQP_GRAD_TYPES_H

// Derivative data
typedef struct {
  cpg_int      init;       // Initialization flag
  cpg_int      *a;         // Bound indicator (-1 for lower bound, 1 for upper bound, 0 for no bound)
  cpg_int      *etree;     // elimination tree
  cpg_int      *Lnz;       // number of nonzeros in each column of L
  cpg_int      *iwork;     // integer workspace
  cpg_int      *bwork;     // boolean workspace
  cpg_float    *fwork;     // float workspace
  cpg_csc      *L;         // Lower triangular factor of K
  cpg_int      *Lmask;     // Boolean mask for fast nonzero querying
  cpg_float    *D;         // Diagonal factor of K
  cpg_float    *Dinv;      // Inverse of D
  cpg_csc      *K;         // KKT matrix
  cpg_float    *c;         // Vector used in update
  cpg_float    *w;         // Vector used in update
  cpg_int      *wi;        // Sparse vector used in update
  cpg_float    *l;         // Vector used in update
  cpg_int      *li;        // Sparse vector used in update
  cpg_float    *lx;        // Sparse vector used in update
  cpg_float    *dx;        // Gradient in x
  cpg_float    *r;         // rhs / solution of linear system
  cpg_float    *dq;        // Gradient in q
  cpg_float    *dl;        // Gradient in l
  cpg_float    *du;        // Gradient in u
  cpg_csc      *dP;        // Gradient in P
  cpg_csc      *dA;        // Gradient in A
} CPG_OSQP_Grad_t;

#endif // ifndef CPG_OSQP_GRAD_TYPES_H

extern CPG_OSQP_Grad_t CPG_OSQP_Grad;