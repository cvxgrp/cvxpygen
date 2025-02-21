
#include "cpg_osqp_grad_workspace.h"

extern void cpg_rank_1_update(cpg_grad_int index, cpg_grad_int n);
extern void cpg_ldl_add(cpg_grad_int index);
extern void cpg_ldl_delete(cpg_grad_int index);
extern void cpg_P_to_K(cpg_grad_csc *P, cpg_grad_csc *K, cpg_grad_csc *K_true);
extern void cpg_A_to_K(cpg_grad_csc *A, cpg_grad_csc *K, cpg_grad_csc *K_true);
extern void cpg_ldl_symbolic();
extern void cpg_ldl_numeric();

extern void cpg_osqp_gradient();