
#include "cpg_workspace.h"

#ifndef CPG_OSQP_GRAD_TYPES_H
# define CPG_OSQP_GRAD_TYPES_H

typedef double cpg_grad_float;
typedef int cpg_grad_int;

typedef struct {
  cpg_grad_int    *p;
  cpg_grad_int    *i;
  cpg_grad_float  *x;
  cpg_grad_int    nnz;
} cpg_grad_csc;

// Derivative data
typedef struct {
  cpg_grad_int      init;       // Initialization flag
  cpg_grad_int      *a;         // Bound indicator (-1 for lower bound, 1 for upper bound, 0 for no bound)
  cpg_grad_int      *etree;     // elimination tree
  cpg_grad_int      *Lnz;       // number of nonzeros in each column of L
  cpg_grad_int      *iwork;     // integer workspace
  cpg_grad_int      *bwork;     // boolean workspace
  cpg_grad_float    *fwork;     // float workspace
  cpg_grad_csc      *L;         // Lower triangular factor of K
  cpg_grad_int      *Lmask;     // Boolean mask for fast nonzero querying
  cpg_grad_float    *D;         // Diagonal factor of K
  cpg_grad_float    *Dinv;      // Inverse of D
  cpg_grad_csc      *K;         // KKT matrix
  cpg_grad_csc      *K_true;    // Exact KKT matrix
  cpg_grad_float    *rhs;       // Right-hand-side
  cpg_grad_float    *delta;     // Vector for iterative refinement
  cpg_grad_float    *c;         // Vector used in update
  cpg_grad_float    *w;         // Vector used in update
  cpg_grad_int      *wi;        // Sparse vector used in update
  cpg_grad_float    *l;         // Vector used in update
  cpg_grad_int      *li;        // Sparse vector used in update
  cpg_grad_float    *lx;        // Sparse vector used in update
  cpg_grad_float    *dx;        // Gradient in x
  cpg_grad_float    *r;         // rhs / solution of linear system
  cpg_grad_float    *dq;        // Gradient in q
  cpg_grad_float    *dl;        // Gradient in l
  cpg_grad_float    *du;        // Gradient in u
  cpg_grad_csc      *dP;        // Gradient in P
  cpg_grad_csc      *dA;        // Gradient in A
} CPG_OSQP_Grad_t;

#endif // ifndef CPG_OSQP_GRAD_TYPES_H

extern CPG_OSQP_Grad_t $workspace$;