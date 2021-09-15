
// map user-defined to OSQP-accepted parameters
extern void canonicalize_params();

// initialize all OSQP-accepted parameters
extern void init_params();

// update OSQP-accepted parameters that depend on user-defined parameters
extern void update_params();

// retrieve user-defined objective function value
extern void retrieve_value();

// retrieve solution in terms of user-defined variables
extern void retrieve_solution();

// perform one ASA sequence to solve a problem instance
extern void solve();
