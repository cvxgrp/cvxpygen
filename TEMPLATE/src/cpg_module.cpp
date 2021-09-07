#include <pybind11/pybind11.h>
#include "cpg_workspace.h"
extern "C" {
    #include "cpg_solve.h"
}

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

void run_example() {
// initialize user-defined parameter values
CPG_Params.delta[0] = 0.30793495262497083687;
CPG_Params.F[0] = 0.51939147930530071839;
CPG_Params.F[1] = 0.78922073617085974906;
CPG_Params.F[2] = 0.18792139162206888958;
CPG_Params.F[3] = 0.76829766153797340866;
CPG_Params.F[4] = 0.87056206247582346158;
CPG_Params.F[5] = 0.26950524561691824310;
CPG_Params.g[0] = 0.49619214098786124101;
CPG_Params.g[1] = 0.73912174711371070313;
CPG_Params.g[2] = 0.19495199164164056516;
CPG_Params.e[0] = 0.17974524672591196683;
CPG_Params.e[1] = 0.53882625857515331624;

// initialize OSQP-accepted parameter values, this must be done once before solving for the first time
init_params();

// solve the problem instance
solve();

// printing objective function value for demonstration purpose
printf("f = %f \n", objective_value[0]);

// printing solution for demonstration purpose
for(int i = 0; i < 2; i++) {
printf("x[%d] = %f \n", i, x[i]);
}
for(int i = 0; i < 2; i++) {
printf("y[%d] = %f \n", i, y[i]);
}
}

namespace py = pybind11;

PYBIND11_MODULE(cpg_module, m) {
    m.doc() = R"pbdoc(
        Pybind11 plugin
    )pbdoc";

    m.def("run_example", &run_example, R"pbdoc(
        Run example.

        Some other explanation.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
