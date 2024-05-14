
#include <math.h>
#include "qdldl.h"
#include "cpg_osqp_grad_workspace.h"

const cpg_int n = $n$;
const cpg_int N = $N$;
cpg_float cpg_grad_a, cpg_grad_a_bar, cpg_grad_gamma;


void cpg_rank_1_update(cpg_int sigma, cpg_int offset) {
    
    cpg_int i, j;
    
    cpg_grad_a = 1.0;

    // Perform rank-1 update in place
    for (j = offset; j < N; j++) {
        cpg_grad_a_bar = cpg_grad_a + sigma * CPG_OSQP_Grad.w[j] * CPG_OSQP_Grad.w[j] * CPG_OSQP_Grad.Dinv[j];
        cpg_grad_gamma = CPG_OSQP_Grad.w[j] * CPG_OSQP_Grad.Dinv[j] / cpg_grad_a_bar;
        CPG_OSQP_Grad.D[j] *= cpg_grad_a_bar / cpg_grad_a;
        CPG_OSQP_Grad.Dinv[j] = 1.0 / CPG_OSQP_Grad.D[j];
        cpg_grad_a = cpg_grad_a_bar;
        for (i = j + 1; i < N; i++) {
            CPG_OSQP_Grad.w[i] -= CPG_OSQP_Grad.w[j] * CPG_OSQP_Grad.L->x[CPG_OSQP_Grad.L->p[j] + i - j - 1];
            CPG_OSQP_Grad.L->x[CPG_OSQP_Grad.L->p[j] + i - j - 1] += sigma * cpg_grad_gamma * CPG_OSQP_Grad.w[i];
        }
    }

}


void cpg_ldl_delete(cpg_int index) {

    cpg_int i, j;

    // Set w
    for (i = 0; i < N - index - 1; i++) {
        CPG_OSQP_Grad.w[index + 1 + i] = CPG_OSQP_Grad.L->x[CPG_OSQP_Grad.L->p[index] + i] * sqrt(-CPG_OSQP_Grad.D[index]);
    }

    // Set index-th row and column of L to zero
    for (i = 0; i < index; i++) {
        CPG_OSQP_Grad.L->x[CPG_OSQP_Grad.L->p[i] + index - i - 1] = 0.0;
    }
    for (i = 0; i < N - index - 1; i++) {
        CPG_OSQP_Grad.L->x[CPG_OSQP_Grad.L->p[index] + i] = 0.0;
    }

    // Set (index, index)-th entry of L and D to 1.0
    //CPG_OSQP_Grad.L->[index + k] = 1.0;
    CPG_OSQP_Grad.D[index] = 1.0;
    CPG_OSQP_Grad.Dinv[index] = 1.0;

    // Update lower right part
    cpg_rank_1_update(-1, index + 1);

}


void cpg_ldl_add(cpg_int index) {

    cpg_int i, j, k;

    // Solve upper left triangular system
    for (i = 0; i < N; i++) {
        CPG_OSQP_Grad.c[i] = CPG_OSQP_Grad.K[i + index * N];
    }
    for (i = 0; i < index; i++) {
        CPG_OSQP_Grad.c[i] *= CPG_OSQP_Grad.Dinv[i];
        for (j = i + 1; j < index; j++) {
            CPG_OSQP_Grad.c[j] -= CPG_OSQP_Grad.L->x[CPG_OSQP_Grad.L->p[i] + j - i - 1] * CPG_OSQP_Grad.D[i] * CPG_OSQP_Grad.c[i];
        }
    }

    // Udpate D and L, first part
    CPG_OSQP_Grad.D[index] = CPG_OSQP_Grad.c[index];
    for (i = 0; i < index; i++) {
        CPG_OSQP_Grad.D[index] -= CPG_OSQP_Grad.c[i] * CPG_OSQP_Grad.c[i] * CPG_OSQP_Grad.D[i];
    }
    CPG_OSQP_Grad.Dinv[index] = 1.0 / CPG_OSQP_Grad.D[index];
    for (i = 0; i < index; i++) {
        CPG_OSQP_Grad.L->x[CPG_OSQP_Grad.L->p[i] + index - i - 1] = CPG_OSQP_Grad.c[i];
    }
    k = index * (2 * N - 1 - index) / 2;
    for (i = index + 1; i < N; i++) {
        k = CPG_OSQP_Grad.L->p[index] + i - index - 1;
        CPG_OSQP_Grad.L->x[k] = CPG_OSQP_Grad.c[i];
        for (j = 0; j < index; j++) {
            CPG_OSQP_Grad.L->x[k] -= CPG_OSQP_Grad.L->x[CPG_OSQP_Grad.L->p[j] + i - j - 1] * CPG_OSQP_Grad.D[j] * CPG_OSQP_Grad.c[j];
        }
        CPG_OSQP_Grad.L->x[k] *= CPG_OSQP_Grad.Dinv[index];
    }

    // Set w
    for (i = 0; i < N - index - 1; i++) {
        CPG_OSQP_Grad.w[index + 1 + i] = CPG_OSQP_Grad.L->x[CPG_OSQP_Grad.L->p[index] + i] * sqrt(-CPG_OSQP_Grad.D[index]);
    }

    // Update D and L, second part
    cpg_rank_1_update(1, index + 1);

}


void cpg_osqp_gradient() {

    cpg_int i, j, k;

    // Check active constraints
    for (i = 0; i < N - n; i++) {
        if (sol_y[i] < -1e-12) {
            if (CPG_OSQP_Grad.a[i] == 0) {
                cpg_ldl_add(n + i);
                CPG_OSQP_Grad.a[i] = -1; // lower bound active
            }
        } else if (sol_y[i] > 1e-12) {
            if (CPG_OSQP_Grad.a[i] == 0) {
                cpg_ldl_add(n + i);
                CPG_OSQP_Grad.a[i] = 1; // upper bound active
            }
        } else {
            if (CPG_OSQP_Grad.a[i] != 0) {
                cpg_ldl_delete(n + i);
                CPG_OSQP_Grad.a[i] = 0; // no bound active
            }
        }
    }

    // Fill rhs of linear system and solve with QDLDL
    for (i = 0; i < n; i++) {
        CPG_OSQP_Grad.r[i] = CPG_OSQP_Grad.dx[i];
    }
    for (i = n; i < N; i++) {
        CPG_OSQP_Grad.r[i] = 0.0;
    }
    QDLDL_solve(N, CPG_OSQP_Grad.L->p, CPG_OSQP_Grad.L->i, CPG_OSQP_Grad.L->x, CPG_OSQP_Grad.Dinv, CPG_OSQP_Grad.r);

    // Fill gradient in q
    for (i = 0; i < n; i++) {
        CPG_OSQP_Grad.dq[i] = - CPG_OSQP_Grad.r[i];
    }

    // Fill gradient in l and u
    for (i = 0; i < N - n; i++) {
        if (CPG_OSQP_Grad.a[i] == -1) {
            CPG_OSQP_Grad.dl[i] = CPG_OSQP_Grad.r[n + i];
            CPG_OSQP_Grad.du[i] = 0.0;
        } else if (CPG_OSQP_Grad.a[i] == 1) {
            CPG_OSQP_Grad.dl[i] = 0.0;
            CPG_OSQP_Grad.du[i] = CPG_OSQP_Grad.r[n + i];
        } else {
            CPG_OSQP_Grad.dl[i] = 0.0;
            CPG_OSQP_Grad.du[i] = 0.0;
        }
    }

    // Fill gradient in P
    for (j = 0; j < n; j++) {
        for (k = CPG_OSQP_Grad.dP->p[j]; k < CPG_OSQP_Grad.dP->p[j + 1]; k++) {
            i = CPG_OSQP_Grad.dP->i[k];
            CPG_OSQP_Grad.dP->x[k] = -0.5 * (CPG_OSQP_Grad.r[i] * sol_x[j] + sol_x[i] * CPG_OSQP_Grad.r[j]);
        }
    }

    // Fill gradient in A
    for (j = 0; j < n; j++) {
        for (k = CPG_OSQP_Grad.dA->p[j]; k < CPG_OSQP_Grad.dA->p[j + 1]; k++) {
            i = CPG_OSQP_Grad.dA->i[k];
            if (CPG_OSQP_Grad.a[i] == 0) {
                CPG_OSQP_Grad.dA->x[k] = 0.0;
            } else {
                CPG_OSQP_Grad.dA->x[k] = - (CPG_OSQP_Grad.r[n + i] * sol_x[j] + sol_y[i] * CPG_OSQP_Grad.r[j]);
            }
        }
    }

}
