
#include <math.h>
#include "qdldl.h"
#include "cpg_osqp_grad_workspace.h"

const cpg_int n = $n$;
const cpg_int N = $N$;
cpg_float cpg_grad_a, cpg_grad_a_bar, cpg_grad_gamma;


// Function to insert a non-zero entry into the sparse matrix
void insert_nonzero(cpg_csc *L, cpg_int row, cpg_int col) {
    if (row < col) return; // Ensure lower triangular property

    cpg_int start = L->p[col];
    cpg_int end = L->p[col + 1];
    cpg_int insert_pos = end;
    
    // Binary search for the row within the column
    while (start < end) {
        cpg_int mid = start + (end - start) / 2;
        if (L->i[mid] == row) {
            return; // Non-zero already exists, no need to insert
        } else if (L->i[mid] < row) {
            start = mid + 1;
        } else {
            end = mid;
            insert_pos = mid; // Potential insertion position
        }
    }

    // Insert the new non-zero entry
    L->nnz++;

    // Move existing elements one step forward to make space for the new one
    for (cpg_int p = L->nnz - 1; p > insert_pos; p--) {
        L->i[p] = L->i[p - 1];
        L->x[p] = L->x[p - 1];
    }

    // Insert the new row index
    L->i[insert_pos] = row;
    L->x[insert_pos] = 0.0;

    // Update the column pointers
    for (cpg_int j = col + 1; j <= N; j++) {
        L->p[j]++;
    }
}

// Function to insert a non-zero entry into the sparse matrix
void insert_nonzero_and_value(cpg_csc *L, cpg_int row, cpg_int col, cpg_float value) {
    if (row < col) return; // Ensure lower triangular property

    cpg_int start = L->p[col];
    cpg_int end = L->p[col + 1];
    cpg_int insert_pos = end;
    
    // Binary search for the row within the column
    while (start < end) {
        cpg_int mid = start + (end - start) / 2;
        if (L->i[mid] == row) {
            L->x[mid] = value;
            return; // Non-zero already exists, no need to insert
        } else if (L->i[mid] < row) {
            start = mid + 1;
        } else {
            end = mid;
            insert_pos = mid; // Potential insertion position
        }
    }

    // Insert the new non-zero entry
    L->nnz++;

    // Move existing elements one step forward to make space for the new one
    for (cpg_int p = L->nnz - 1; p > insert_pos; p--) {
        L->i[p] = L->i[p - 1];
        L->x[p] = L->x[p - 1];
    }

    // Insert the new row index
    L->i[insert_pos] = row;
    L->x[insert_pos] = value;

    // Update the column pointers
    for (cpg_int j = col + 1; j <= N; j++) {
        L->p[j]++;
    }
}

// Function to remove a non-zero entry from the sparse matrix
void remove_nonzero(cpg_csc *L, cpg_int row, cpg_int col) {
    if (row < col) return; // Ensure lower triangular property

    // Check if the entry exists and find its position
    cpg_int remove_pos = -1;
    for (cpg_int p = L->p[col]; p < L->p[col + 1]; p++) {
        if (L->i[p] == row) {
            if (fabs(L->x[p]) > 1e-6) {
                return; // Entry is numerically non-zero, do not remove
            }
            remove_pos = p;
            break;
        }
    }

    if (remove_pos == -1) {
        return; // Entry does not exist, nothing to remove
    }

    // Shift elements to remove the entry
    for (cpg_int p = remove_pos; p < L->nnz - 1; p++) {
        L->i[p] = L->i[p + 1];
        L->x[p] = L->x[p + 1];
    }

    // Decrease the number of non-zero elements
    L->nnz--;

    // Update the column pointers
    for (cpg_int j = col + 1; j <= N; j++) {
        L->p[j]--;
    }
}

// Function to perform symbolic update
void symbolic_ldl_update(cpg_csc *L, cpg_int *w_indices, cpg_int w_size, cpg_int offset) {
    for (cpg_int k = 0; k < w_size; k++) {
        cpg_int col = w_indices[k] + offset;
        for (cpg_int j = k + 1; j < w_size; j++) {
            cpg_int row = w_indices[j] + offset;
            insert_nonzero(L, row, col);
        }
    }
}

// Function to perform symbolic downdate
void symbolic_ldl_downdate(cpg_csc *L, cpg_int *w_indices, cpg_int w_size, cpg_int offset) {
    for (cpg_int k = 0; k < w_size; k++) {
        cpg_int col = w_indices[k] + offset;
        for (cpg_int j = k + 1; j < w_size; j++) {
            cpg_int row = w_indices[j] + offset;
            remove_nonzero(L, row, col);
        }
    }
}


void cpg_rank_1_update(cpg_int sigma, cpg_int offset) {
    
    cpg_int i, j, k;
    
    cpg_grad_a = 1.0;

    // Perform rank-1 update in place
    for (j = offset; j < N; j++) {
        cpg_grad_a_bar = cpg_grad_a + sigma * CPG_OSQP_Grad.w[j] * CPG_OSQP_Grad.w[j] / CPG_OSQP_Grad.D[j];
        cpg_grad_gamma = CPG_OSQP_Grad.w[j] / (CPG_OSQP_Grad.D[j] * cpg_grad_a_bar);
        CPG_OSQP_Grad.D[j] *= cpg_grad_a_bar / cpg_grad_a;
        CPG_OSQP_Grad.Dinv[j] = 1.0 / CPG_OSQP_Grad.D[j];
        cpg_grad_a = cpg_grad_a_bar;
        for (k = CPG_OSQP_Grad.L->p[j]; k < CPG_OSQP_Grad.L->p[j + 1]; k++) {
            i = CPG_OSQP_Grad.L->i[k];
            if (i > j) {
                CPG_OSQP_Grad.w[i] -= CPG_OSQP_Grad.w[j] * CPG_OSQP_Grad.L->x[k];
                CPG_OSQP_Grad.L->x[k] += sigma * cpg_grad_gamma * CPG_OSQP_Grad.w[i];
            }
        }
    }

}


void cpg_ldl_delete(cpg_int index) {

    cpg_int i, j, k;

    // Set w
    for (i = 0; i < N; i++) {
        CPG_OSQP_Grad.w[i] = 0.0;
    }
    cpg_int wsize = CPG_OSQP_Grad.L->p[index + 1] - CPG_OSQP_Grad.L->p[index];
    for (i = 0; i < wsize; i++) {
        CPG_OSQP_Grad.wi[i] = CPG_OSQP_Grad.L->i[CPG_OSQP_Grad.L->p[index] + i] - index - 1; // w indices are local
        CPG_OSQP_Grad.w[index + 1 + CPG_OSQP_Grad.wi[i]] = CPG_OSQP_Grad.L->x[CPG_OSQP_Grad.L->p[index] + i] * sqrt(-CPG_OSQP_Grad.D[index]);
    }

    // Set index-th row and column of L to zero
    for (j = 0; j < index; j++) {
        for (k = CPG_OSQP_Grad.L->p[j]; k < CPG_OSQP_Grad.L->p[j + 1]; k++) {
            i = CPG_OSQP_Grad.L->i[k];
            if (i >= index) {
                if (i == index) {
                    CPG_OSQP_Grad.L->x[k] = 0.0;
                }
                break;
            }
        }
    }
    for (k = CPG_OSQP_Grad.L->p[index]; k < CPG_OSQP_Grad.L->p[index + 1]; k++) {
        CPG_OSQP_Grad.L->x[k] = 0.0;
    }
    
    // Set index-th entry of D to -1.0
    CPG_OSQP_Grad.D[index] = -1.0;
    CPG_OSQP_Grad.Dinv[index] = -1.0;

    // Perform symbolic update
    symbolic_ldl_update(CPG_OSQP_Grad.L, CPG_OSQP_Grad.wi, wsize, index + 1);

    // Update lower right part
    cpg_rank_1_update(-1, index + 1);

}


void cpg_ldl_add(cpg_int index) {

    cpg_int i, j, k, r;

    // Solve upper left triangular system

    // fill index-th (partial) column of K (starting from 0 to index-1) into c
    for (i = 0; i < N; i++) CPG_OSQP_Grad.c[i] = 0.0;
    cpg_int c_size = 0;
    for (k = CPG_OSQP_Grad.K->p[index]; k < CPG_OSQP_Grad.K->p[index + 1] - 1; k++) {
        i = CPG_OSQP_Grad.K->i[k];
        CPG_OSQP_Grad.c[i] = CPG_OSQP_Grad.K->x[k];
        c_size++;
    }

    // solve L (Dl) = c
    for (i = 0; i < index; i++) CPG_OSQP_Grad.l[i] = 0.0;
    for (j = 0; j < index; j++) {
        CPG_OSQP_Grad.l[j] += CPG_OSQP_Grad.c[j];
        for (k = CPG_OSQP_Grad.L->p[j]; k < CPG_OSQP_Grad.L->p[j + 1]; k++) {
            CPG_OSQP_Grad.l[CPG_OSQP_Grad.L->i[k]] -= CPG_OSQP_Grad.L->x[k] * CPG_OSQP_Grad.l[j];
        }
    }

    // Solve Dl = l
    cpg_int l_size = 0;
    for (i = 0; i < index; i++) {
        if (CPG_OSQP_Grad.l[i] != 0.0) {
            CPG_OSQP_Grad.lx[l_size] = CPG_OSQP_Grad.l[i] / CPG_OSQP_Grad.D[i];
            CPG_OSQP_Grad.li[l_size] = i;
            l_size++;
        }
    }

    // Udpate D and L, first part
    // get K[index, index]
    CPG_OSQP_Grad.D[index] = CPG_OSQP_Grad.K->x[CPG_OSQP_Grad.K->p[index + 1] - 1];
    for (k = 0; k < l_size; k++) {
        CPG_OSQP_Grad.D[index] -= CPG_OSQP_Grad.lx[k] * CPG_OSQP_Grad.lx[k] * CPG_OSQP_Grad.D[CPG_OSQP_Grad.li[k]];
    }
    CPG_OSQP_Grad.Dinv[index] = 1.0 / CPG_OSQP_Grad.D[index];
    // fill index-th row of L with l
    for (k = 0; k < l_size; k++) {
        insert_nonzero_and_value(CPG_OSQP_Grad.L, index, CPG_OSQP_Grad.li[k], CPG_OSQP_Grad.lx[k]);
    }

    // Use dense storage l to store values of l32, while l12 is still stored in sparse li and lx
    for (i = 0; i < N; i++) CPG_OSQP_Grad.l[i] = 0.0;
    for (j = index + 1; j < N; j++) {
        for (k = CPG_OSQP_Grad.K->p[j]; k < CPG_OSQP_Grad.K->p[j + 1]; k++) {
            i = CPG_OSQP_Grad.K->i[k];
            if (i >= index) {
                if (i == index) {
                    CPG_OSQP_Grad.l[j - index - 1] = CPG_OSQP_Grad.K->x[k];
                }
                break;
            }
        }
    }
    for (r = 0; r < l_size; r++) {
        for (k = CPG_OSQP_Grad.L->p[CPG_OSQP_Grad.li[r]]; k < CPG_OSQP_Grad.L->p[CPG_OSQP_Grad.li[r] + 1]; k++) {
            i = CPG_OSQP_Grad.L->i[k];
            if (i > index) {
                CPG_OSQP_Grad.l[i - index - 1] -= CPG_OSQP_Grad.L->x[k] * CPG_OSQP_Grad.lx[r] * CPG_OSQP_Grad.D[CPG_OSQP_Grad.li[r]];
            }
        }
    }

    // overwrite l12 with l32 in lx and li
    l_size = 0;
    for (i = 0; i < N - index - 1; i++) {
        if (CPG_OSQP_Grad.l[i] != 0.0) {
            CPG_OSQP_Grad.lx[l_size] = CPG_OSQP_Grad.l[i] / CPG_OSQP_Grad.D[index];
            CPG_OSQP_Grad.li[l_size] = i;
            l_size++;
        }
    }

    // fill l32 into L
    for (k = 0; k < l_size; k++) {
        insert_nonzero_and_value(CPG_OSQP_Grad.L, CPG_OSQP_Grad.li[k] + index + 1, index, CPG_OSQP_Grad.lx[k]);
    }

    // Set w
    for (i = 0; i < N; i++) {
        CPG_OSQP_Grad.w[i] = 0.0;
    }
    cpg_int wsize = l_size;
    for (i = 0; i < wsize; i++) {
        CPG_OSQP_Grad.wi[i] = CPG_OSQP_Grad.li[i]; // w indices are local
        CPG_OSQP_Grad.w[index + 1 + CPG_OSQP_Grad.wi[i]] = CPG_OSQP_Grad.lx[i] * sqrt(-CPG_OSQP_Grad.D[index]);
    }

    // Update D and L, second part
    cpg_rank_1_update(1, index + 1);

    // Perform symbolic update
    //symbolic_ldl_downdate(CPG_OSQP_Grad.L, CPG_OSQP_Grad.wi, wsize, index + 1);

}

void cpg_ldl_symbolic() {

    cpg_int nnz = QDLDL_etree(
        N,
        CPG_OSQP_Grad.K->p, CPG_OSQP_Grad.K->i,
        CPG_OSQP_Grad.iwork, CPG_OSQP_Grad.Lnz, CPG_OSQP_Grad.etree
    );
    CPG_OSQP_Grad.L->nnz = nnz;

}

void cpg_ldl_numeric() {

    QDLDL_factor(
        N,
        CPG_OSQP_Grad.K->p, CPG_OSQP_Grad.K->i, CPG_OSQP_Grad.K->x,
        CPG_OSQP_Grad.L->p, CPG_OSQP_Grad.L->i, CPG_OSQP_Grad.L->x,
        CPG_OSQP_Grad.D, CPG_OSQP_Grad.Dinv,
        CPG_OSQP_Grad.Lnz, CPG_OSQP_Grad.etree, CPG_OSQP_Grad.bwork, CPG_OSQP_Grad.iwork, CPG_OSQP_Grad.fwork
    );

}

void cpg_P_to_K(cpg_csc *P, cpg_csc *K) {

    for (cpg_int col = 0; col < n; col++) {
        cpg_int col_start = P->p[col];
        cpg_int col_end = P->p[col + 1];
        
        for (cpg_int idx = col_start; idx < col_end; idx++) {
            cpg_int row = P->i[idx];
            cpg_float value = P->x[idx];
            
            // Upper left block insertion
            cpg_int upper_left_row = row;
            cpg_int upper_left_col = col;
            for (cpg_int k_idx = K->p[upper_left_col]; k_idx < K->p[upper_left_col + 1]; k_idx++) {
                if (K->i[k_idx] == upper_left_row) {
                    K->x[k_idx] = value;
                    if (upper_left_row == upper_left_col) {
                        K->x[k_idx] += 1e-6;
                    }
                    break;
                }
            }
        }
    }

}

void cpg_A_to_K(cpg_csc *A, cpg_csc *K) {

    // Fill A (csc format) into lower left block and transpose into upper right block of K (column-major)
    for (cpg_int col = 0; col < n; col++) {
        // Get the start and end pointers for the column in A
        cpg_int col_start = A->p[col];
        cpg_int col_end = A->p[col + 1];
        
        // Iterate over each non-zero entry in the column
        for (cpg_int idx = col_start; idx < col_end; idx++) {
            cpg_int row = A->i[idx];      // Row index in A
            cpg_float value = A->x[idx];  // Value in A
            
            // Compute indices for K            
            cpg_int upper_right_row = col;    // Upper right block: row in K
            cpg_int upper_right_col = n + row; // Upper right block: col in K
            
            // Find the position in K for upper right block
            cpg_int k_upper_start = K->p[upper_right_col];
            cpg_int k_upper_end = K->p[upper_right_col + 1];
            for (cpg_int k_idx = k_upper_start; k_idx < k_upper_end; k_idx++) {
                if (K->i[k_idx] == upper_right_row) {
                    K->x[k_idx] = value;
                    break;
                }
            }
        }
    }

}


void cpg_osqp_gradient() {

    cpg_int i, j, k;

    // Check active constraints
    for (i = 0; i < N - n; i++) {
        if (sol_y[i] < -1e-6) {
            if (CPG_OSQP_Grad.a[i] == 0) {
                cpg_ldl_add(n + i);
                CPG_OSQP_Grad.a[i] = -1; // lower bound active
            }
        } else if (sol_y[i] > 1e-6) {
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
