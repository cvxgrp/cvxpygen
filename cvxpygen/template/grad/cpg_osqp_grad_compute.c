
#include <math.h>
#include "qdldl.h"
#include "cpg_osqp_grad_workspace.h"

const cpg_grad_int n = $n$;
const cpg_grad_int N = $N$;
cpg_grad_float cpg_grad_a, cpg_grad_a_bar, cpg_grad_gamma;


// Function to insert a non-zero entry into the sparse matrix
void insert_nonzero(cpg_grad_csc *L, cpg_grad_int row, cpg_grad_int col) {
    if (row < col) return; // Ensure lower triangular property

    cpg_grad_int start = L->p[col];
    cpg_grad_int end = L->p[col + 1];
    cpg_grad_int insert_pos = end;
    
    // Binary search for the row within the column
    while (start < end) {
        cpg_grad_int mid = start + (end - start) / 2;
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
    for (cpg_grad_int p = L->nnz - 1; p > insert_pos; p--) {
        L->i[p] = L->i[p - 1];
        L->x[p] = L->x[p - 1];
    }

    // Insert the new row index
    L->i[insert_pos] = row;
    L->x[insert_pos] = 0.0;

    // Update the column pointers
    for (cpg_grad_int j = col + 1; j <= N; j++) {
        L->p[j]++;
    }
}

// Function to insert a non-zero entry into the sparse matrix
void insert_nonzero_and_value(cpg_grad_csc *L, cpg_grad_int row, cpg_grad_int col, cpg_grad_float value) {
    if (row < col) return; // Ensure lower triangular property

    cpg_grad_int start = L->p[col];
    cpg_grad_int end = L->p[col + 1];
    cpg_grad_int insert_pos = end;
    
    // Binary search for the row within the column
    while (start < end) {
        cpg_grad_int mid = start + (end - start) / 2;
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
    for (cpg_grad_int p = L->nnz - 1; p > insert_pos; p--) {
        L->i[p] = L->i[p - 1];
        L->x[p] = L->x[p - 1];
    }

    // Insert the new row index
    L->i[insert_pos] = row;
    L->x[insert_pos] = value;

    // Update the column pointers
    for (cpg_grad_int j = col + 1; j <= N; j++) {
        L->p[j]++;
    }
}

// Function to remove a non-zero entry from the sparse matrix
void remove_nonzero(cpg_grad_csc *L, cpg_grad_int row, cpg_grad_int col) {
    if (row < col) return; // Ensure lower triangular property

    // Check if the entry exists and find its position
    cpg_grad_int remove_pos = -1;
    for (cpg_grad_int p = L->p[col]; p < L->p[col + 1]; p++) {
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
    for (cpg_grad_int p = remove_pos; p < L->nnz - 1; p++) {
        L->i[p] = L->i[p + 1];
        L->x[p] = L->x[p + 1];
    }

    // Decrease the number of non-zero elements
    L->nnz--;

    // Update the column pointers
    for (cpg_grad_int j = col + 1; j <= N; j++) {
        L->p[j]--;
    }
}

// Function to perform symbolic update
void symbolic_ldl_update(cpg_grad_csc *L, cpg_grad_int *w_indices, cpg_grad_int w_size, cpg_grad_int offset) {
    for (cpg_grad_int k = 0; k < w_size; k++) {
        cpg_grad_int col = w_indices[k] + offset;
        for (cpg_grad_int j = k + 1; j < w_size; j++) {
            cpg_grad_int row = w_indices[j] + offset;
            cpg_grad_int ind = (2*N-3-col)*col/2+row-1;
            if ($workspace$.Lmask[ind] == 0) {
                $workspace$.Lmask[ind] = 1;
                insert_nonzero(L, row, col);
            } 
        }
    }
}

// Function to perform symbolic downdate
void symbolic_ldl_downdate(cpg_grad_csc *L, cpg_grad_int *w_indices, cpg_grad_int w_size, cpg_grad_int offset) {
    for (cpg_grad_int k = 0; k < w_size; k++) {
        cpg_grad_int col = w_indices[k] + offset;
        for (cpg_grad_int j = k + 1; j < w_size; j++) {
            cpg_grad_int row = w_indices[j] + offset;
            remove_nonzero(L, row, col);
        }
    }
}


void cpg_rank_1_update(cpg_grad_int sigma, cpg_grad_int offset) {
    
    cpg_grad_int i, j, k;
    
    cpg_grad_a = 1.0;

    // Perform rank-1 update in place
    for (j = offset; j < N; j++) {
        cpg_grad_a_bar = cpg_grad_a + sigma * $workspace$.w[j] * $workspace$.w[j] / $workspace$.D[j];
        cpg_grad_gamma = $workspace$.w[j] / ($workspace$.D[j] * cpg_grad_a_bar);
        $workspace$.D[j] *= cpg_grad_a_bar / cpg_grad_a;
        $workspace$.Dinv[j] = 1.0 / $workspace$.D[j];
        cpg_grad_a = cpg_grad_a_bar;
        for (k = $workspace$.L->p[j]; k < $workspace$.L->p[j + 1]; k++) {
            i = $workspace$.L->i[k];
            if (i > j) {
                $workspace$.w[i] -= $workspace$.w[j] * $workspace$.L->x[k];
                $workspace$.L->x[k] += sigma * cpg_grad_gamma * $workspace$.w[i];
            }
        }
    }

}


void cpg_ldl_delete(cpg_grad_int index) {

    cpg_grad_int i, j, k;

    // Set w
    for (i = 0; i < N; i++) {
        $workspace$.w[i] = 0.0;
    }
    cpg_grad_int wsize = $workspace$.L->p[index + 1] - $workspace$.L->p[index];
    for (i = 0; i < wsize; i++) {
        $workspace$.wi[i] = $workspace$.L->i[$workspace$.L->p[index] + i] - index - 1; // w indices are local
        $workspace$.w[index + 1 + $workspace$.wi[i]] = $workspace$.L->x[$workspace$.L->p[index] + i] * sqrt(-$workspace$.D[index]);
    }

    // Set index-th row and column of L to zero
    for (j = 0; j < index; j++) {
        for (k = $workspace$.L->p[j]; k < $workspace$.L->p[j + 1]; k++) {
            i = $workspace$.L->i[k];
            if (i >= index) {
                if (i == index) {
                    $workspace$.L->x[k] = 0.0;
                }
                break;
            }
        }
    }
    for (k = $workspace$.L->p[index]; k < $workspace$.L->p[index + 1]; k++) {
        $workspace$.L->x[k] = 0.0;
    }
    
    // Set index-th entry of D to -1.0
    $workspace$.D[index] = -1.0;
    $workspace$.Dinv[index] = -1.0;

    // Perform symbolic update
    symbolic_ldl_update($workspace$.L, $workspace$.wi, wsize, index + 1);

    // Update lower right part
    cpg_rank_1_update(-1, index + 1);

}


void cpg_ldl_add(cpg_grad_int index) {

    cpg_grad_int i, j, k, r;

    // Solve upper left triangular system

    // fill index-th (partial) column of K (starting from 0 to index-1) into c
    for (i = 0; i < N; i++) $workspace$.c[i] = 0.0;
    cpg_grad_int c_size = 0;
    for (k = $workspace$.K->p[index]; k < $workspace$.K->p[index + 1] - 1; k++) {
        i = $workspace$.K->i[k];
        $workspace$.c[i] = $workspace$.K->x[k];
        c_size++;
    }

    // solve L (Dl) = c
    for (i = 0; i < index; i++) $workspace$.l[i] = 0.0;
    for (j = 0; j < index; j++) {
        $workspace$.l[j] += $workspace$.c[j];
        for (k = $workspace$.L->p[j]; k < $workspace$.L->p[j + 1]; k++) {
            $workspace$.l[$workspace$.L->i[k]] -= $workspace$.L->x[k] * $workspace$.l[j];
        }
    }

    // Solve Dl = l
    cpg_grad_int l_size = 0;
    for (i = 0; i < index; i++) {
        if ($workspace$.l[i] != 0.0) {
            $workspace$.lx[l_size] = $workspace$.l[i] / $workspace$.D[i];
            $workspace$.li[l_size] = i;
            l_size++;
        }
    }

    // Udpate D and L, first part
    // get K[index, index]
    $workspace$.D[index] = $workspace$.K->x[$workspace$.K->p[index + 1] - 1];
    for (k = 0; k < l_size; k++) {
        $workspace$.D[index] -= $workspace$.lx[k] * $workspace$.lx[k] * $workspace$.D[$workspace$.li[k]];
    }
    $workspace$.Dinv[index] = 1.0 / $workspace$.D[index];
    // fill index-th row of L with l
    for (k = 0; k < l_size; k++) {
        insert_nonzero_and_value($workspace$.L, index, $workspace$.li[k], $workspace$.lx[k]);
    }

    // Use dense storage l to store values of l32, while l12 is still stored in sparse li and lx
    for (i = 0; i < N; i++) $workspace$.l[i] = 0.0;
    for (j = index + 1; j < N; j++) {
        for (k = $workspace$.K->p[j]; k < $workspace$.K->p[j + 1]; k++) {
            i = $workspace$.K->i[k];
            if (i >= index) {
                if (i == index) {
                    $workspace$.l[j - index - 1] = $workspace$.K->x[k];
                }
                break;
            }
        }
    }
    for (r = 0; r < l_size; r++) {
        for (k = $workspace$.L->p[$workspace$.li[r]]; k < $workspace$.L->p[$workspace$.li[r] + 1]; k++) {
            i = $workspace$.L->i[k];
            if (i > index) {
                $workspace$.l[i - index - 1] -= $workspace$.L->x[k] * $workspace$.lx[r] * $workspace$.D[$workspace$.li[r]];
            }
        }
    }

    // overwrite l12 with l32 in lx and li
    l_size = 0;
    for (i = 0; i < N - index - 1; i++) {
        if ($workspace$.l[i] != 0.0) {
            $workspace$.lx[l_size] = $workspace$.l[i] / $workspace$.D[index];
            $workspace$.li[l_size] = i;
            l_size++;
        }
    }

    // fill l32 into L
    for (k = 0; k < l_size; k++) {
        insert_nonzero_and_value($workspace$.L, $workspace$.li[k] + index + 1, index, $workspace$.lx[k]);
    }

    // Set w
    for (i = 0; i < N; i++) {
        $workspace$.w[i] = 0.0;
    }
    cpg_grad_int wsize = l_size;
    for (i = 0; i < wsize; i++) {
        $workspace$.wi[i] = $workspace$.li[i]; // w indices are local
        $workspace$.w[index + 1 + $workspace$.wi[i]] = $workspace$.lx[i] * sqrt(-$workspace$.D[index]);
    }

    // Update D and L, second part
    cpg_rank_1_update(1, index + 1);

    // Perform symbolic update
    //symbolic_ldl_downdate($workspace$.L, $workspace$.wi, wsize, index + 1);

}

void cpg_ldl_symbolic() {

    cpg_grad_int nnz = QDLDL_etree(
        N,
        $workspace$.K->p, $workspace$.K->i,
        $workspace$.iwork, $workspace$.Lnz, $workspace$.etree
    );
    $workspace$.L->nnz = nnz;

}

void cpg_ldl_numeric() {

    QDLDL_factor(
        N,
        $workspace$.K->p, $workspace$.K->i, $workspace$.K->x,
        $workspace$.L->p, $workspace$.L->i, $workspace$.L->x,
        $workspace$.D, $workspace$.Dinv,
        $workspace$.Lnz, $workspace$.etree, $workspace$.bwork, $workspace$.iwork, $workspace$.fwork
    );

}

void cpg_P_to_K(cpg_grad_csc *P, cpg_grad_csc *K, cpg_grad_csc *K_true) {

    for (cpg_grad_int col = 0; col < n; col++) {

        for (cpg_grad_int idx = P->p[col]; idx < P->p[col + 1]; idx++) {

            cpg_grad_int row = P->i[idx];
            cpg_grad_float value = P->x[idx];

            // Upper left block insertion into K
            for (cpg_grad_int k_idx = K->p[col]; k_idx < K->p[col + 1]; k_idx++) {
                if (K->i[k_idx] == row) {
                    K->x[k_idx] = value;
                    if (row == col) {
                        K->x[k_idx] += 1e-6;
                    }
                    break;
                }
            }

            // Upper left block insertion into K_true
            for (cpg_grad_int k_idx = K_true->p[row]; k_idx < K_true->p[row + 1]; k_idx++) {
                if (K_true->i[k_idx] == col) {
                    K_true->x[k_idx] = value;
                    break;
                }
            }

        }

    }

}

void cpg_A_to_K(cpg_grad_csc *A, cpg_grad_csc *K, cpg_grad_csc *K_true) {

    // Fill A (csc format) into lower left block and transpose into upper right block of K (column-major)
    for (cpg_grad_int col = 0; col < n; col++) {
        
        // Iterate over each non-zero entry in the column
        for (cpg_grad_int idx = A->p[col]; idx < A->p[col + 1]; idx++) {
            cpg_grad_int row = A->i[idx];      // Row index in A
            cpg_grad_float value = A->x[idx];  // Value in A
            
            // Compute indices for K            
            cpg_grad_int upper_right_row = col;    // Upper right block: row in K
            cpg_grad_int upper_right_col = n + row; // Upper right block: col in K

            // Compute indices for lower left block
            cpg_grad_int lower_left_row = n + row;    // Lower left block: row in K
            cpg_grad_int lower_left_col = col;    // Lower left block: col in K
            
            // Find the position in K for upper right block
            for (cpg_grad_int k_idx = K->p[upper_right_col]; k_idx < K->p[upper_right_col + 1]; k_idx++) {
                if (K->i[k_idx] == upper_right_row) {
                    K->x[k_idx] = value;
                    break;
                }
            }

            // Find the position in K_true (csr) for upper right block
            for (cpg_grad_int k_idx = K_true->p[upper_right_row]; k_idx < K_true->p[upper_right_row + 1]; k_idx++) {
                if (K_true->i[k_idx] == upper_right_col) {
                    K_true->x[k_idx] = value;
                    break;
                }
            }

            // Find the position in K_true (csr) for lower left block
            for (cpg_grad_int k_idx = K_true->p[lower_left_row]; k_idx < K_true->p[lower_left_row + 1]; k_idx++) {
                if (K_true->i[k_idx] == lower_left_col) {
                    K_true->x[k_idx] = value;
                    break;
                }
            }

        }

    }

}


void cpg_osqp_gradient() {

    cpg_grad_int i, j, k, l;

    // Check active constraints
    for (i = 0; i < N - n; i++) {
        if (sol_y[i] < -1e-12) {
            if ($workspace$.a[i] == 0) {
                cpg_ldl_add(n + i);
                $workspace$.a[i] = -1; // lower bound active
            }
        } else if (sol_y[i] > 1e-12) {
            if ($workspace$.a[i] == 0) {
                cpg_ldl_add(n + i);
                $workspace$.a[i] = 1; // upper bound active
            }
        } else {
            if ($workspace$.a[i] != 0) {
                cpg_ldl_delete(n + i);
                $workspace$.a[i] = 0; // no bound active
            }
        }
    }

    // Fill rhs of linear system and solve with QDLDL
    for (i = 0; i < n; i++) {
        $workspace$.r[i] = $workspace$.dx[i];
        $workspace$.rhs[i] = $workspace$.dx[i];
    }
    for (i = n; i < N; i++) {
        $workspace$.r[i] = 0.0;
        $workspace$.rhs[i] = 0.0;
    }
    QDLDL_solve(N, $workspace$.L->p, $workspace$.L->i, $workspace$.L->x, $workspace$.Dinv, $workspace$.r);
    
    // Three iterations of iterative refinement
    for (l = 0; l < 3; l++) {
        // Compute delta = rhs - K_true @ r
        for (i = 0; i < N; i++) {
            $workspace$.delta[i] = $workspace$.rhs[i];
        }
        for (i = 0; i < N; i++) {
            if (i >= n && $workspace$.a[i - n] == 0) {
                $workspace$.delta[i] = 0.0;
                continue;
            }
            for (k = $workspace$.K_true->p[i]; k < $workspace$.K_true->p[i + 1]; k++) {
                j = $workspace$.K_true->i[k];
                if (j >= n && $workspace$.a[j - n] == 0) continue;
                $workspace$.delta[i] -= $workspace$.K_true->x[k] * $workspace$.r[j];
            }
        }
        // Solve K @ delta = rhs - K_true @ r
        QDLDL_solve(N, $workspace$.L->p, $workspace$.L->i, $workspace$.L->x, $workspace$.Dinv, $workspace$.delta);
        // Update r
        for (i = 0; i < N; i++) {
            $workspace$.r[i] += $workspace$.delta[i];
        }
    }

    // Fill gradient in q
    for (i = 0; i < n; i++) {
        $workspace$.dq[i] = - $workspace$.r[i];
    }

    // Fill gradient in l and u
    for (i = 0; i < N - n; i++) {
        if ($workspace$.a[i] == -1) {
            $workspace$.dl[i] = $workspace$.r[n + i];
            $workspace$.du[i] = 0.0;
        } else if ($workspace$.a[i] == 1) {
            $workspace$.dl[i] = 0.0;
            $workspace$.du[i] = $workspace$.r[n + i];
        } else {
            $workspace$.dl[i] = 0.0;
            $workspace$.du[i] = 0.0;
        }
    }

    // Fill gradient in P
    for (j = 0; j < n; j++) {
        for (k = $workspace$.dP->p[j]; k < $workspace$.dP->p[j + 1]; k++) {
            i = $workspace$.dP->i[k];
            $workspace$.dP->x[k] = -0.5 * ($workspace$.r[i] * sol_x[j] + sol_x[i] * $workspace$.r[j]);
        }
    }

    // Fill gradient in A
    for (j = 0; j < n; j++) {
        for (k = $workspace$.dA->p[j]; k < $workspace$.dA->p[j + 1]; k++) {
            i = $workspace$.dA->i[k];
            if ($workspace$.a[i] == 0) {
                $workspace$.dA->x[k] = 0.0;
            } else {
                $workspace$.dA->x[k] = - ($workspace$.r[n + i] * sol_x[j] + sol_y[i] * $workspace$.r[j]);
            }
        }
    }

}
