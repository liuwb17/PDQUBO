/// @file bin_packing_easy.cpp
/// @brief This file contains an example of solving the bin packing problem
/// using the EasySolver.
/// @details
/// The bin packing problem is a combinatorial optimization problem that is
/// concerned with packing items of different sizes into bins of fixed size in a
/// way that minimizes the number of bins used. In this example, we consider a
/// bin packing problem with 4 bins and 8 items. The weights of the items are
/// given as {33, 61, 58, 41, 50, 21, 60, 64}. The goal is to find a way to pack
/// the items into the bins such that the total weight of items in each bin is
/// at most 120.
/// @author Koji Nakano
/// @copyright Copyright (c) 2024, Koji Nakano
/// @version 2024-01-05

#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"

int main() {
  // The number of bins
  const size_t bin_count = 4;
  // The weights of items
  qbpp::Vector<int> weights = {33, 61, 58, 41, 50, 21, 60, 64};

  // Define a matrix of binary variables
  // x[i][j] = 1 if item j is in bin i, 0 otherwise
  auto x = qbpp::var("x", bin_count, weights.size());

  // Each item is in exactly one bin, i.e., the sum of each column is 1
  auto f = qbpp::sum(qbpp::vector_sum(qbpp::transpose(x)) == 1);

  // The total weight of items in each bin is at most 120
  for (size_t i = 0; i < bin_count; i++) {
    f += 0 <= qbpp::sum(x[i] * weights) <= 120;
  }

  f.simplify_as_binary();

  std::cout << "f = " << f << std::endl;

  auto solver = qbpp::easy_solver::EasySolver(f);

  solver.set_target_energy(0);
  solver.enable_default_callback();

  auto sol = solver.search();

  std::cout << "Solution = " << sol << std::endl;

  for (size_t i = 0; i < bin_count; i++) {
    std::cout << "Bin " << i << " :";
    for (size_t j = 0; j < weights.size(); j++) {
      if (sol.get(x[i][j]) == 1) {
        std::cout << " " << weights[j];
      }
    }
    std::cout << std::endl;
  }
}