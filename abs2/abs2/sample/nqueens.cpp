#include "abs2.hpp"

// Class to store QUBO model for NQUEENS problem.
class NQUEENS_Model : public abs2::Model {
  // dim x dim chess board
  const int dim;

private:
  // Set the value for ((i1,j1),(i2,j2)) of NQUEENS QUBO model.
  void set(int x1, int y1, int x2, int y2, int val) {
    abs2::Model::set(x1 * dim + y1, x2 * dim + y2, val);
  }

public:
  // Creating QUBO model for NQUEENS
  NQUEENS_Model(int dim) : abs2::Model(dim * dim, 8), dim(dim) {
    for (int x = 0; x < dim; ++x)
      for (int y = 0; y < dim; ++y) {
        set(x, y, x, y, -1);
        for (int i = x + 1; i < dim; ++i)
          set(x, y, i, y, +1);
        for (int j = y + 1; j < dim; ++j)
          set(x, y, x, j, +1);
        for (int d = 1; d < dim - x; ++d) {
          if (y - d >= 0)
            set(x, y, x + d, y - d, +1);
          if (y + d < dim)
            set(x, y, x + d, y + d, +1);
        }
      }
  }
};

int main(int argc, char *argv[]) {
  int dim = 32;

  if (argc > 1) {
    dim = std::atoi(argv[1]);
  }

  // Create and initializes the ABS2 QUBO solver.
  abs2::Solver solver;
  // Generate a QUBO model for N-Queens for dim x dim chess board.
  NQUEENS_Model model(dim);
  // Create "param" to store parameters.
  abs2::Param param;
  // Set "time_limit" to "10" seconds.
  param.set("time_limit", "60");
  // Set "target_energy" to -dim seconds.
  param.set("target_energy", std::to_string(-dim));
  // Solve "model" with "param" using "solver".
  auto sol = solver(model, param);
  // Print "sol".
  for (int x = 0; x < dim; ++x) {
    for (int y = 0; y < dim; ++y)
      std::cout << sol.get(x * dim + y);
    std::cout << "\n";
  }
  // Print "sol" sttributes
  sol.print("attrs");
}
