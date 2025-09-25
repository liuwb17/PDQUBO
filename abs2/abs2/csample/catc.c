#include "abs2c.h"
#include "string.h"

abs2CallbackPtr cb;

void mycallback(const char *event) {
  if (strcmp(event, "init") == 0) {
    // set callback option to call callback when new solution is found.
    abs2Callback_set(cb, "new");
  } else if (strcmp(event, "new") == 0) {
    // print solution if new solution is found
    abs2SolPtr sol = abs2Callback_get(cb);
    abs2Sol_print(sol);
  }
}

int main(int argc, char *argv[]) {
  // Create ABS2 QUBO Solver
  abs2SolverPtr solver = abs2Solver_gen();
  // Load QUBO model from the file "cat.mm.gz".
  abs2ModelPtr model = abs2Model_load("../data/cat.mm.gz");
  // Create "param" to store parameters.
  abs2ParamPtr param = abs2Param_gen();
  // Set "time_limit" to 100 seconds.
  abs2Param_set(param, "time_limit", "100");
  // Set appropriate search parameters for "cat.mm.gz".
  abs2Param_set(param, "search_cycles", "500 100");
  abs2Param_set(param, "best_priority", "10");
  // Set callback function mycallback to cb
  cb = abs2Param_callback(param, mycallback);
  // Create "sol" to store the solution.
  abs2SolPtr sol = abs2Sol_gen(0);
  // Solve "model" with "param" and store the solution in "sol".
  abs2Solve(solver, sol, model, param);
  // Print the solution "sol".
  abs2Sol_print(sol);
}