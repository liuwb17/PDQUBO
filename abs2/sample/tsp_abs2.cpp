/// @file tsp_abs2.cpp
/// @author Koji Nakano
/// @brief Solves randomly generated Traveling Salesman Problem (TSP) using the
/// ABS2 QUBO solver.
/// @version 2024-09-16

#include <boost/program_options.hpp>

#include "qbpp.hpp"
#include "qbpp_abs2.hpp"
#include "qbpp_misc.hpp"
#include "qbpp_tsp.hpp"

namespace po = boost::program_options;

/// @brief Class to define ABS2 callback function for factorization.
class ABS2Callback : public qbpp_abs2::Callback {
  /// @brief The TSP expression.
  const qbpp::tsp::TSPQuadModel &tsp_quad_model;

 public:
  /// @brief Construct a new ABS2 callback object
  /// @param tsp_quad_model The TSP QuadModel.
  ABS2Callback(const qbpp::tsp::TSPQuadModel &tsp_quad_model)
      : qbpp_abs2::Callback(tsp_quad_model), tsp_quad_model(tsp_quad_model) {}

  /// @brief Callback function for ABS2 solver.
  /// @param event The event name.
  void callback(const std::string &event) override {
    if (event == "init") {
      // Enable the callback for every new best solution.
      set("new");
    } else if (event == "new") {
      auto sol = get_sol();
      qbpp::tsp::TSPSol tsp_sol(tsp_quad_model, sol);
      std::cout << "TTS = " << std::fixed << std::setprecision(3)
                << sol.get_tts() << "s ";
      tsp_sol.print();
    }
  }
};

/// @brief Main function to generate a random map and solve the Traveling
/// Salesman Problem (TSP) using the ABS2 solver.
/// @param argc Number of command-line arguments.
/// @param argv List of command-line arguments.
/// @details Generates a random map with the specified number of nodes and
/// random seed, and solves it using the ABS2 solver.
int main(int argc, char **argv) {
  // clang-format off
  po::options_description desc(
      "Solving the randomly generated TSP using QUBO++ Easy Solver");
  desc.add_options()("help,h", "produce help message")
     ("nodes,n", po::value<uint32_t>()->default_value(10),"set the number of nodes in the TSP map")
     ("time,t", po::value<uint32_t>()->default_value(10), "set time limit in seconds")
     ("tsp_seed,s", po::value<uint32_t>(), "set the random seed for the TSP map")
     ("output,o", po::value<std::string>(), "set the output file (png, svg, etc) to save the TSP solution")
     ("fix,f", "fix node 0 as the starting node");
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

  /// @brief The size of the input set
  uint32_t nodes = vm["nodes"].as<uint32_t>();
  /// @brief The time limit for the solver
  uint32_t time_limit = vm["time"].as<uint32_t>();
  /// @brief True if node 0 is fixed as the starting node
  bool fix_first = vm.count("fix");

  // Set the random seed for deterministic behavior if seed is provided.
  if (vm.count("tsp_seed")) {
    qbpp::misc::RandomGenerator::set_seed(vm["tsp_seed"].as<uint32_t>());
  } else {
    qbpp::misc::RandomGenerator::rd_seed();
  }

  std::cout << "Generating random TSP Map with " << nodes << " nodes"
            << std::endl;
  qbpp::tsp::TSPMap tsp_map;
  tsp_map.gen_random_map(nodes);

  std::cout << "Generating a TSP QUBO expression" << std::endl;
  qbpp::tsp::TSPQuadModel tsp_quad_model(tsp_map, fix_first);

  std::cout << "Variables = " << tsp_quad_model.var_count()
            << " Linear Terms = " << tsp_quad_model.term_count(1)
            << " Quadratic Terms = " << tsp_quad_model.term_count(2)
            << std::endl;

  // Initialize ABS2 solver
  qbpp_abs2::Solver abs2_solver;
  // Create a model for ABS2 solver from model
  qbpp_abs2::QuadModel abs2_model(tsp_quad_model);
  // Create a callback function for ABS2 solver.
  ABS2Callback abs2_callback(tsp_quad_model);
  // Create a parameters object to store solver parameters.
  qbpp_abs2::Param abs2_param;
  // Set a time limit.
  abs2_param.set_time_limit(time_limit);
  // Set the callback function for ABS2 solver.
  abs2_param.set(abs2_callback);

  std::cout << "Solving the TSP" << std::endl;
  auto sol = abs2_solver(abs2_model, abs2_param);

  qbpp::tsp::TSPSol tsp_sol(tsp_quad_model, sol);
  tsp_sol.print();

  if (vm.count("output")) {
    qbpp::tsp::DrawSimpleGraph graph;
    for (uint32_t i = 0; i < nodes; ++i) graph.add_node(tsp_map[i]);

    for (uint32_t i = 0; i < nodes; ++i)
      graph.add_edge(tsp_sol[i], tsp_sol[(i + 1) % nodes]);

    graph.draw(vm["output"].as<std::string>());
  }
}
