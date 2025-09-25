/// @file knapsack_exhaustive.cpp
/// @brief This file contains an example of solving the knapsack problem using
/// the exhaustive solver.
/// @author Koji Nakano
/// @copyright Copyright (c) 2024, Koji Nakano
/// @version 2025-01-05

#include "qbpp.hpp"
#include "qbpp_exhaustive_solver.hpp"

int main() {
  // The values and weights of items.
  qbpp::Vector<int> values = {20, 30, 50, 60, 40, 10, 25, 35};
  qbpp::Vector<int> weights = {4, 5, 8, 7, 6, 3, 4, 5};

  // Define a vector of binary variables.
  // x[i] = 1 if item i is selected, 0 otherwise.
  auto x = qbpp::var("x", values.size());

  // Define the objective function, which is to maximize the total value of
  // selected items.
  auto objective = qbpp::sum(x * values);

  // Define the constraint, which is to limit the total weight of selected items
  // to at most 20.
  auto constraint = 0 <= qbpp::sum(x * weights) <= 20;

  // Define the QUBO expression.
  auto f = -objective + constraint * 1000;

  f.simplify_as_binary();

  std::cout << "f = " << f << std::endl;

  auto solver = qbpp::exhaustive_solver::ExhaustiveSolver(f);

  auto sol = solver.search();

  std::cout << "Solution = " << sol << std::endl;

  for (size_t i = 0; i < values.size(); i++) {
    if (sol.get(x[i]) == 1) {
      std::cout << "Item " << i << " with value " << values[i] << " and weight "
                << weights[i] << " is selected" << std::endl;
    }
  }
}