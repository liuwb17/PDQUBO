#include "abs2c.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // Create ABS2 QUBO Solver
  abs2SolverPtr solver = abs2Solver_gen();
  // Create QUBO "model" of 32 binary variables and 8-bit coefficients.
  abs2ModelPtr model = abs2Model_gen(32, 8);
  // Set a random integer from -10 to 10 to each W_{i,j}.
  for (int i = 0; i < 32; ++i)
    for (int j = i; j < 32; ++j)
      abs2Model_set(model, i, j, rand() % 21 - 10);
  // print the generated QUBO model
  abs2Model_print(model);
  // Create "param" to store parameters.
  abs2ParamPtr param = abs2Param_gen();
  // Set "time_limit" to 10 seconds.
  abs2Param_set(param, "time_limit", "10");
  abs2SolPtr sol = abs2Sol_gen(0);
  // Solve "model" with "param" and store the solution in "sol".
  abs2Solve(solver, sol, model, param);
  // Print the solution "sol".
  abs2Sol_print(sol);
}