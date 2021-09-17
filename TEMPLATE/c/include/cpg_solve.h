
// map user-defined to OSQP-accepted parameters
extern void canonicalize_OSQP_P();
extern void canonicalize_OSQP_q();
extern void canonicalize_OSQP_d();
extern void canonicalize_OSQP_A();
extern void canonicalize_OSQP_l();
extern void canonicalize_OSQP_u();

// retrieve user-defined objective function value
extern void retrieve_value();

// retrieve solution in terms of user-defined variables
extern void retrieve_solution();

// perform one ASA sequence to solve a problem instance
extern void solve();

// update user-defined parameter values
