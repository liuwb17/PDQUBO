#include "abs2c.h"
#include "string.h"
#include <stdio.h>
#include <unistd.h>

abs2CallbackPtr cb;

void mycallback(const char *event) {
  // will callback in 2 secconds.
  abs2Callback_set_operand(cb, "alarm", "2");
  if (strcmp(event, "init") == 0) {
    // set callback option to call callback when new solution is found.
    abs2Callback_set(cb, "new");
  } else {
    abs2Sol_print(abs2Callback_get(cb));
  }
}

int main(int argc, char *argv[]) {
  // Create ABS2 QUBO Solver
  abs2SolverPtr solver = abs2Solver_gen();
  // Load QUBO model from the file "nug30.mm.gz".
  abs2ModelPtr model = abs2Model_load("../data/nug30.mm.gz");
  // Create "param" to store parameters.
  abs2ParamPtr param = abs2Param_gen();
  // Set "time_limit" to 10 seconds.
  abs2Param_set(param, "time_limit", "10");
  // Create "hint" to store solution.
  abs2SolPtr hint = abs2Sol_gen(0);
  // Solve "model" with "param" and store the solution in "hint".
  abs2Solve(solver, hint, model, param);
  // Print the solution "hint".
  abs2Sol_print(hint);
  // Set call back function
  cb = abs2Param_callback(param, mycallback);
  // Set "time_limit" to 10 seconds.
  abs2Param_set(param, "time_limit", "30");
  // Create "sol" to store the solution.
  abs2SolPtr sol = abs2Sol_gen(0);
  // Solve "model" with "param" and "hint" and store the solution in "sol".
  abs2Solve_start(solver, sol, model, param, hint);
  // Print the solution "sol".
  abs2Sol_print(sol);
}