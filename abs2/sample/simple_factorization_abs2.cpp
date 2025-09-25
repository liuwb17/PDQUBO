/// @file simple_factorization_abs2.cpp
/// @brief Simple factorization example using ABS2 QUBO Solver
/// @details This is a simple example of factorization using ABS2 QUBO Solver.
/// It solves the equation x * y = 97 * 89.
/// @author Koji Nakano
/// @version 2024-10-05

#include "qbpp.hpp"
#include "qbpp_abs2.hpp"

int main() {
  auto x = 1 <= qbpp::var_int("x") <= 100;
  auto y = 1 <= qbpp::var_int("y") <= 100;
  auto f = x * y == 97 * 89;
  auto quad_model = qbpp::QuadModel(simplify_as_binary(reduce(f)));
  std::cout << "quad_model has " << quad_model.var_count() << " variables, "
            << quad_model.term_count(1) << " linear terms, and "
            << quad_model.term_count(2) << " quadratic terms" << std::endl;
  auto solver = qbpp_abs2::Solver();
  auto param = qbpp_abs2::Param();
  auto callback = qbpp_abs2::Callback(quad_model);
  param.set_time_limit(10);
  param.set_target_energy(0);
  param.set(callback);
  auto sol = solver(quad_model, param);
  std::cout << "x = " << x << " = " << sol.get(x) << std::endl;
  std::cout << "y = " << y << " = " << sol.get(y) << std::endl;
}