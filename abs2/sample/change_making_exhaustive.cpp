/// @file change_making_exhaustive.cpp
/// @details This sample program demonstrates how to solve a variation of the
/// change-making problem: Given coins of denominations 1, 5, 10, and 25,
/// determine the minimum number of coins needed to achieve a specified amount.
/// This program uses the qbpp library to formulate the problem as a QUBO.
/// @author Koji Nakano
/// @version 2025-01-06

#include "qbpp.hpp"
#include "qbpp_exhaustive_solver.hpp"

int main() {
  int c;
  std::cout << "Enter the amount (non-negative integer): ";
  std::cin >> c;

  // Define variables for each coin type
  auto p = 0 <= qbpp::var_int("p") <= c;
  auto n = 0 <= qbpp::var_int("n") <= c / 5;
  auto d = 0 <= qbpp::var_int("d") <= c / 10;
  auto q = 0 <= qbpp::var_int("q") <= c / 25;

  // Objective: Minimize the total number of coins
  auto obj = p + n + d + q;

  // Constraint: Total value of coins equals the c
  auto total = p + 5 * n + 10 * d + 25 * q;
  auto constraint = (total == c);

  // QUBO formulation
  auto f = obj + constraint * c;

  // Simplify the QUBO expression
  f.simplify_as_binary();

  // Define the solver
  qbpp::exhaustive_solver::ExhaustiveSolver solver(f);

  // Search for all optimal solutions
  auto sol = solver.search();

  // Print the solution
  std::cout << "f = " << f << std::endl;
  std::cout << "Optimal solution found: " << sol << std::endl;
  std::cout << "Total number of coins: " << sol.get(obj) << std::endl;
  std::cout << "Details:\n";
  std::cout << "  Penny(1): " << sol.get(p) << std::endl;
  std::cout << "  Nickel(5): " << sol.get(n) << std::endl;
  std::cout << "  Dime(10): " << sol.get(d) << std::endl;
  std::cout << "  Quarter(25): " << sol.get(q) << std::endl;
}