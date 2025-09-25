#include "abs2c.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  abs2SolverPtr solver = abs2Solver_gen();
  printf("solver = %s\n", abs2Solver_get_attr(solver, "solver"));
  printf("version = %s\n", abs2Solver_get_attr(solver, "version"));
  printf("license = %s\n", abs2Solver_get_attr(solver, "license"));
  printf("max_size = %s\n", abs2Solver_get_attr(solver, "max_size"));
  printf("max_bits = %s\n", abs2Solver_get_attr(solver, "max_bits"));
}