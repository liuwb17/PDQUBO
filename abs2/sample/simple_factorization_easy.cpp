/// @file simple_factorization_easy.cpp
/// @brief Simple factorization example using EasySolver
/// @details This is a simple example of factorization using EasySolver.
/// It solves the equation x * y = 97 * 89.
/// @author Koji Nakano
/// @version 2024-10-05

#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"

int main() {
  auto x = 1 <= qbpp::var_int("x") <= 100;
  auto y = 1 <= qbpp::var_int("y") <= 100;
  auto f = x * y == 97 * 89;
  auto quad_model = qbpp::QuadModel(simplify_as_binary(reduce(f)));
  std::cout << "Quad_model has " << quad_model.var_count() << " variables, "
            << quad_model.term_count(1) << " linear terms, and "
            << quad_model.term_count(2) << " quadratic terms" << std::endl;
  auto solver = qbpp::easy_solver::EasySolver(quad_model);
  solver.set_time_limit(10);
  solver.set_target_energy(0);
  solver.enable_default_callback();
  auto sol = solver.search();
  std::cout << "x = " << x << " = " << sol.get(x) << std::endl;
  std::cout << "y = " << y << " = " << sol.get(y) << std::endl;
}