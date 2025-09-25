#include "abs2.hpp"

int main(int argc, char *argv[]) {
  // Create QUBO "model" of 32 binary variables and 8-bit coefficients.
  abs2::Model model(32, 8);
  // Set a random integer from -10 to 10 to each W_{i,j}.
  for (int i = 0; i < 32; ++i)
    for (int j = i; j < 32; ++j)
      model.set(i, j, rand() % 21 - 10);
  // Print the generated QUBO model.
  model.print();
  // Create "param" to store parameters.
  abs2::Param param;
  // Set "time_limit" to "10" seconds.
  param.set("time_limit", "10");
  // Create and initializes the ABS2 QUBO solver.
  abs2::Solver solver;
  // Solve "model" with "param" and store the solution in "sol".
  auto sol = solver(model, param);
  // Print the solution "sol".
  sol.print();
}