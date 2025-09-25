
/// @file qbpp_partition.hpp
/// @brief Generates QUBO models for Partitioning problem
/// @author Koji Nakano
/// @version 2024-07-08

#ifndef QBPP_PARTITION_HPP
#define QBPP_PARTITION_HPP
#include <numeric>
#include <vector>

#include "qbpp.hpp"

namespace qbpp {
/// @brief Namespace for sample programs using QUBO++ library
namespace sample {
/// @brief Returns a QUBO expression for the partitioning problem
/// @param input_integers A list of integers to be partitioned
/// @return A QUBO expression for the partitioning problem
qbpp::QuadModel partition(const std::vector<int> &input_integers) {
  // Variables for the partitioning problem.
  auto x = qbpp::var("x", input_integers.size());
  //
  int64_t sum =
      std::accumulate(input_integers.begin(), input_integers.end(), 0);
  qbpp::Expr expr;
  // The expression to compute the sum of integers with x[i] = 1 times 2.
  for (size_t i = 0; i < input_integers.size(); i++) {
    expr += 2 * input_integers[i] * x[i];
  }
  // The QuadModel for the partitioning problem
  return qbpp::QuadModel(simplify_as_binary(qbpp::sqr(expr - sum)));
}

}  // namespace sample
}  // namespace qbpp

#endif  // QBPP_PARTITION_HPP