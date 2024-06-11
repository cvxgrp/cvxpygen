
#include "cpg_osqp_grad_workspace.h"

extern void cpg_rank_1_update(cpg_int index, cpg_int n);
extern void cpg_ldl_add(cpg_int index);
extern void cpg_ldl_delete(cpg_int index);
extern void cpg_P_to_K(cpg_csc *P);
extern void cpg_A_to_K(cpg_csc *A);
extern void cpg_ldl();

extern void cpg_osqp_gradient();