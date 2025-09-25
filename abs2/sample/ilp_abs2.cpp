/// @file ilp_abs2.cpp
/// @brief Solves an Integer Linear Programming (ILP) problem using ABS2 solver
/// through QUBO++ library.
/// @details
/// This is a sample program to solve the following Integer Linear Programming
/// (ILP) problem using EasySolver through QUBO++ library.
/// @code
/// Maximize
///   2 * x + 3 * y
/// Subject To
///   c1: x + y <= 7
///   c2: 3 * x + 5 * y <= 28
/// Bounds
///   x <= 7
///   y <= 5
/// Generals x y End
/// @endcode
/// @author Koji Nakano
/// @copyright Copyright (c) 2025, Koji Nakano
/// @version 2025-02-04

#include <iostream>

#include "qbpp.hpp"
#include "qbpp_abs2.hpp"

int main() {
  auto x = 0 <= qbpp::var_int("x") <= 7;
  auto y = 0 <= qbpp::var_int("y") <= 5;
  auto obj = 2 * x + 3 * y;
  auto c1 = 0 <= x + y <= 7;
  auto c2 = 0 <= 3 * x + 5 * y <= 28;

  auto f = -obj + 100 * (c1 + c2);
  f.simplify_as_binary();
  std::cout << "f = " << f << std::endl;

  auto quad_model = qbpp_abs2::QuadModel(f);

  // Define the solver
  auto solver = qbpp_abs2::Solver();

  // Define the parameters
  auto param = qbpp_abs2::Param();

  // Set the time limit
  param.set_time_limit(5);

  // Create a default callback function
  qbpp_abs2::Callback callback(quad_model);

  // Set the callback function
  param.set(callback);

  // Solve the ILP problem
  auto sol = solver(quad_model, param);

  std::cout << sol << std::endl;
  std::cout << "x = " << sol.get(x) << std::endl;
  std::cout << "y = " << sol.get(y) << std::endl;
  std::cout << "obj = " << sol.get(obj) << std::endl;
  std::cout << "*c1 = " << sol.get(*c1) << std::endl;
  std::cout << "*c2 = " << sol.get(*c2) << std::endl;
}
