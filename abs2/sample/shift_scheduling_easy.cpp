/// @file shift_scheduling_easy.cpp
/// @brief Solving shift scheduling problem with Easy Solver
/// The shift scheduling problem is to assign workers to continuous time slots
/// of a site.
/// @author Koji Nakano
/// @version 2025-02-06

#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"

int main() {
  // Number of sites
  const size_t s = 3;
  // Required workers in each time slot of each site.
  const std::vector<int> required_workers = {1, 1, 2, 2, 2, 2,
                                             2, 2, 2, 2, 1, 1};
  // minimum and maximum of workers working hours
  const std::vector<std::pair<int, int>> workers = {
      {0, 4}, {0, 4}, {4, 5}, {4, 5}, {4, 5}, {5, 6}, {5, 6},
      {5, 6}, {6, 7}, {6, 7}, {7, 8}, {7, 8}, {7, 8}};
  // Number of workers
  const size_t m = workers.size();
  // Number of time slots
  const size_t n = required_workers.size();

  // Define QUBO variables
  // x[i][j][k] = 1 if worker j works at time slot k in site i;
  auto x = qbpp::var("x", s, m, n);

  // Total number of working hours
  auto total = qbpp::total_sum(x);

  // worker_working_hours[j] = total number of working hours of worker j
  auto worker_working_hours = qbpp::expr(m);
  for (size_t j = 0; j < m; j++) {
    for (size_t i = 0; i < s; i++) {
      for (size_t k = 0; k < n; k++) {
        worker_working_hours[j] += x[i][j][k];
      }
    }
  }

  // time_slot_workers[i][k] = total number of workers at time slot k in site i
  auto time_slot_workers = qbpp::expr(s, n);
  for (size_t i = 0; i < s; i++) {
    for (size_t k = 0; k < n; k++) {
      for (size_t j = 0; j < m; j++) {
        time_slot_workers[i][k] += x[i][j][k];
      }
    }
  }

  // worker_diff[j] = -2 if worker j do not work.
  // worker_diff[j] = 0 if worker j works continuously.
  // worker_diff[j] >= 2 if worker j works discontinuously.
  auto worker_diff = qbpp::expr(m);
  for (size_t j = 0; j < m; j++) {
    for (size_t i = 0; i < s; i++) {
      for (size_t k = 0; k < n - 1; k++) {
        worker_diff[j] += qbpp::sqr(x[i][j][k] - x[i][j][k + 1]);
      }
      worker_diff[j] +=
          qbpp::sqr(0 - x[i][j][0]) + qbpp::sqr(x[i][j][n - 1] - 0);
    }
    worker_diff[j] -= 2;
  }

  auto constraint = qbpp::expr();

  // constraints for the working hours of each worker
  for (size_t j = 0; j < m; j++) {
    // priority for worker_working hours for worker_diff
    // Since all zero in worker_diff takes value 0, while continuous 1s
    // take value 2, we need to set the priority larger than 2 when the minimum
    // working hours is 1.
    qbpp::coeff_t first = workers[j].first;
    qbpp::coeff_t second = workers[j].second;
    if (first == 0 && second == 0) {
      constraint += worker_working_hours[j];
    } else if (first == second) {
      constraint += worker_working_hours[j] == first;
    } else if (first == 0 && second == 1) {
      constraint += 0 <= worker_working_hours[j] <= 1;
    } else if (first == 0) {  // second >= 2
      constraint +=
          2 * (1 <= worker_working_hours[j] <= second) + worker_diff[j];
    } else if (first == 1 && second == 2) {
      constraint +=
          3 * (first <= worker_working_hours[j] <= second) + worker_diff[j];
    } else if (first == 1) {  // second >= 3
      constraint +=
          2 * (first <= worker_working_hours[j] <= second) + worker_diff[j];
    } else {
      constraint +=
          (first <= worker_working_hours[j] <= second) + worker_diff[j];
    }
  }

  // Each time slot in each site has 1 or 2 workers.
  for (size_t i = 0; i < s; i++) {
    for (size_t k = 0; k < n; k++) {
      constraint += required_workers[k] <= time_slot_workers[i][k] <=
                    required_workers[k] + 1;
    }
  }

  auto f = total + 100 * constraint;

  f.simplify_as_binary();
  auto solver = qbpp::easy_solver::EasySolver(f);
  solver.enable_default_callback();
  solver.set_time_limit(2);
  auto sol = solver.search();
  std::cout << "sol = " << sol << std::endl;

  std::cout << "total = " << sol.get(total) << std::endl;

  for (size_t j = 0; j < m; j++) {
    std::cout << "worker " << std::setw(2) << j << " (" << workers[j].first
              << "," << workers[j].second
              << ") : " << sol.get(worker_working_hours[j]) << " hours ";
    for (size_t i = 0; i < s; i++) {
      for (size_t k = 0; k < n; k++) {
        std::cout << qbpp::str(sol.get(x[i][j][k]));
      }
      std::cout << " ";
    }
    std::cout << std::endl;
  }

  std::cout << std::string(25, ' ');
  for (size_t i = 0; i < s; i++) {
    std::cout << " ";
    for (size_t k = 0; k < n; k++) {
      std::cout << sol.get(time_slot_workers[i][k]);
    }
  }
  std::cout << std::endl;

  std::cout << "required workers" << std::string(9, ' ');
  for (size_t i = 0; i < s; i++) {
    std::cout << " ";
    for (size_t k = 0; k < n; k++) {
      std::cout << required_workers[k];
    }
  }
  std::cout << std::endl;
}
