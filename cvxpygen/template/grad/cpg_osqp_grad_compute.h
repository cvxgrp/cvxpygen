
#include "cpg_osqp_grad_workspace.h"

extern void cpg_rank_1_update(cpg_int index, cpg_int n);
extern void cpg_ldl_add(cpg_int index);
extern void cpg_ldl_delete(cpg_int index);
extern void cpg_P_to_K(cpg_csc *P, cpg_csc *K);
extern void cpg_A_to_K(cpg_csc *A, cpg_csc *K);
extern void cpg_ldl_symbolic();
extern void cpg_ldl_numeric();

extern void cpg_osqp_gradient();