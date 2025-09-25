/// @file simple_shift_scheduling_easy.cpp
/// @brief Solving shift scheduling problem with Easy Solver
/// @details
/// This is a simple example of solving a shift scheduling problem with Easy
/// Solver.
/// The problem is to assign 7 workers to 24 time slots so that the total
/// working hours is minimized.
/// The constraints are as follows:
/// - Each worker works 7 or 8 hours.
/// - Each time slot is worked by 2 or 3 workers.
/// @author Koji Nakano
/// @version 2025-01-18

#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"

int main() {
  // Number of workers
  const size_t m = 7;
  // Number of time slots
  const size_t n = 24;

  // Define QUBO variables
  // x[i][j] = 1 if worker i works at time slot j.
  auto x = qbpp::var("x", m, n);

  // time_slots[j] stores the total number of workers assigned to time slot j.
  auto time_slots = qbpp::vector_sum(qbpp::transpose(x));

  // total stores the total number of working hours.
  auto total = qbpp::total_sum(x);

  // Define constraints: Each time slot is worked by 2 or 3 workers.
  auto constraint = qbpp::sum(2 <= time_slots <= 3);
  // worker[i] stores the total number of hours worked by worker i.
  auto workers = qbpp::vector_sum(x);
  // worker[i] must be 7 or 8.
  constraint += qbpp::sum(7 <= workers <= 8);
  // Each row of x must be consecutive ones.
  constraint += qbpp::sum(qbpp::diff(x));

  auto f = total + 100 * constraint;

  f.simplify_as_binary();
  auto solver = qbpp::easy_solver::EasySolver(f);
  solver.enable_default_callback();
  solver.set_time_limit(1);
  auto sol = solver.search();
  std::cout << "sol = " << sol << std::endl;

  auto workers_sol = qbpp::eval(workers, sol);
  auto time_slots_sol = qbpp::eval(time_slots, sol);

  for (size_t i = 0; i < m; i++) {
    std::cout << "worker " << i << ": " << workers_sol[i] << " hours ";
    for (size_t j = 0; j < n; j++) {
      std::cout << qbpp::str(sol.get(x[i][j]));
    }
    std::cout << std::endl;
  }
  std::cout << "workers           ";
  for (size_t j = 0; j < n; j++) {
    std::cout << time_slots_sol[j];
  }
  std::cout << std::endl;
}