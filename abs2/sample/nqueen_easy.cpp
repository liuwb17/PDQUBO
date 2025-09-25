/// @file nqueen_easy.cpp
/// @author Koji Nakano
/// @brief Solves the N-Queens problem using ABS2 QUBO solver from QUBO++
/// library.
/// @version 2024-11-23

#include <boost/program_options.hpp>

#include "qbpp.hpp"
#include "qbpp_easy_solver.hpp"
#include "qbpp_nqueen.hpp"

namespace po = boost::program_options;

/// @brief Solves the N-Queens problem using EasySolver in the QUBO++ library.
/// library.
/// @param argc Number of command line arguments.
/// @param argv Command line arguments.
/// @return int Return code.
/// @details The dimension of the chessboard can be specified by the command
/// line argument.
int main(int argc, char *argv[]) {
  // clang-format off
  po::options_description desc(
      "N-Queens Problem Solver using QUBO++ Easy Solver");
  desc.add_options()("help,h", "produce help message")
     ("dimension,d", po::value<int>()->default_value(8), "set dimension of the chessboard")
     ("time_limit,t", po::value<int>()->default_value(10), "set time limit in seconds")
     ("seed,s", po::value<int>(), "set random seed")
     ("expand,e", "expand the one-hot formula for QUBO model generation")
     ("fast,f", "fast mode for QUBO model generation")
     ("parallel,p", "parallel mode for QUBO model generation (default)");
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

  int dimension = vm["dimension"].as<int>();
  int time_limit = vm["time_limit"].as<int>();

  // Set the random seed for deterministic behavior if seed is provided.
  if (vm.count("seed")) {
    qbpp::misc::RandomGenerator::set_seed(vm["seed"].as<int>());
  }

  qbpp::nqueen::NQueenQuadModel::Mode mode;
  if (vm.count("expand")) {
    mode = qbpp::nqueen::NQueenQuadModel::Mode::EXPAND;
  } else if (vm.count("fast")) {
    mode = qbpp::nqueen::NQueenQuadModel::Mode::FAST;
  } else if (vm.count("parallel")) {
    mode = qbpp::nqueen::NQueenQuadModel::Mode::PARALLEL;
  } else {
    mode = qbpp::nqueen::NQueenQuadModel::Mode::PARALLEL;
  }

  qbpp::nqueen::NQueenQuadModel nqueen_model(dimension, mode);

  std::cout << "Generating the QUBO model." << std::endl;

  std::cout << "Variables = " << nqueen_model.var_count()
            << " Linear Terms = " << nqueen_model.term_count(1)
            << " Quadratic Terms = " << nqueen_model.term_count(2) << std::endl;

  std::cout << "Generating an EasySolver object for the QUBO model."
            << std::endl;
  auto solver = qbpp::easy_solver::EasySolver(nqueen_model);
  solver.set_time_limit(time_limit);
  solver.set_target_energy(0);
  solver.enable_default_callback();

  std::cout << "Executing the EasySolver to solve the QUBO model." << std::endl;
  auto sol = solver.search();

  // Print the solution as a chessboard.
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j)
      std::cout << static_cast<int>(sol.get(nqueen_model.get_var(i, j)));
    std::cout << std::endl;
  }

  std::cout << "Dimension = " << dimension << " TTS = " << std::fixed
            << std::setprecision(3) << std::setfill('0') << solver.get_tts()
            << "s Energy = " << sol.energy() << std::endl;

  std::cout << "flip_count = " << solver.get_flip_count() << std::endl;
}
