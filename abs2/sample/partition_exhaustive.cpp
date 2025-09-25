/// @file partition_exhaustive.cpp
/// @brief Solves the Partitioning problem using the QUBO++ Exhaustive solver
/// @author Koji Nakano
/// @version 2025-01-04

#include <boost/program_options.hpp>

#include "qbpp.hpp"
#include "qbpp_exhaustive_solver.hpp"
#include "qbpp_misc.hpp"

namespace po = boost::program_options;

/// @brief Solves the Partitioning problem using the QUBO++ Exhaustive Solver
/// @param argc Number of command-line arguments
/// @param argv Command-line arguments
/// @return Exit code
int main(int argc, char **argv) {
  // clang-format off
  po::options_description desc(
      "Solving the random partitioning problem using QUBO++ Exhaustive Solver.");
  desc.add_options()("help,h", "produce help message.")
    ("size,s", po::value<size_t>()->default_value(10), "Set the size of the input set.")
    ("random,r", po::value<uint32_t>(), "Set the base seed for the random number generator for deterministic behavior.")
    ("max,m", po::value<qbpp::energy_t>()->default_value(1000), "Set the maximum value of the input set.");
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  } catch (const std::exception &e) {
    std::cout << "Wrong arguments. Please use -h/--help option to see the "
                 "usage.\n";
    return 1;
  }
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  // Set the random seed for deterministic behavior if seed is provided.
  if (vm.count("random")) {
    qbpp::misc::RandomGenerator::set_seed(vm["random"].as<uint32_t>());
  }

  /// @brief The size of the input set
  size_t size = vm["size"].as<size_t>();
  /// @brief The maximum value of the input set
  qbpp::energy_t max_val = vm["max"].as<qbpp::energy_t>();

  /// @brief A vector to store the input set
  qbpp::Vector<qbpp::energy_t> w;

  /// @brief Generates the input set
  for (size_t i = 0; i < size; i++) {
    w.emplace_back(qbpp::misc::RandomGenerator::gen(max_val));
  }

  /// @brief Prints the input set
  std::cout << qbpp::str(w, "w") << std::endl;

  auto x = qbpp::var("x", w.size());
  auto f = qbpp::sqr(2 * qbpp::sum(w * x) - qbpp::sum(w)).simplify_as_binary();

  std::cout << "f = " << f << std::endl;

  // Generates a QUBO++ easy solver object from the QUBO model
  auto solver = qbpp::exhaustive_solver::ExhaustiveSolver(f);

  // Enables the default callback
  solver.enable_default_callback();

  // Executes the QUBO++ easy solver
  auto sol = solver.search();

  // Prints the QUBO solution
  std::cout << "Solution = " << sol << std::endl;

  // Prints the sum of the input set and the sum of the elements in the set
  std::cout << "sum0 = " << qbpp::eval(qbpp::sum(w * (1 - x)), sol) << " :";
  for (size_t i = 0; i < size; i++) {
    if (sol.get(x[i]) == 0) {
      std::cout << " " << w[i];
    }
  }
  std::cout << std::endl;
  std::cout << "sum1 = " << qbpp::eval(qbpp::sum(w * x), sol) << " :";
  for (size_t i = 0; i < size; i++) {
    if (sol.get(x[i]) == 1) {
      std::cout << " " << w[i];
    }
  }
  std::cout << std::endl;
}