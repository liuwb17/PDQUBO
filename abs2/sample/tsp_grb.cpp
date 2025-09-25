/// @file tsp_grb.cpp
/// @author Koji Nakano
/// @brief Solves randomly generated Traveling Salesman Problem (TSP) using the
/// QUBO++ Easy solver.
/// @version 2024-11-11

#include <boost/program_options.hpp>

#include "qbpp.hpp"
#include "qbpp_grb.hpp"
#include "qbpp_misc.hpp"
#include "qbpp_tsp.hpp"

namespace po = boost::program_options;

/// @brief Class to define Gurobi callback function for factorization.
/// @note This Gurobi callback inherits from qbpp_grb::Callback.
class GRB_Callback : public qbpp_grb::Callback {
  const qbpp::tsp::TSPQuadModel &tsp_quad_model;
  std::optional<qbpp::energy_t> target_energy;

 public:
  /// @brief Construct a new grb callback object
  /// @param tsp_quad_model The TSP expression.
  /// @param target_energy The target energy.
  GRB_Callback(const qbpp::tsp::TSPQuadModel &tsp_quad_model,
               qbpp::energy_t target_energy)
      : qbpp_grb::Callback(tsp_quad_model),
        tsp_quad_model(tsp_quad_model),
        target_energy(target_energy) {}

  /// @brief callback function for Gurobi optimizer.
  /// @details The callback function is called by the Gurobi optimizer for
  /// every event. It displays the solution of the TSP when a new best solution
  /// obtained.
  void callback() override {
    if (where == GRB_CB_MIPSOL) {
      auto sol = get_sol();
      qbpp::tsp::TSPSol tsp_sol(tsp_quad_model, sol);
      std::cout << "TTS = " << std::fixed << std::setprecision(3)
                << std::setfill('0') << qbpp::get_time() << "s ";
      tsp_sol.print();
      if (target_energy.has_value() && sol.energy() <= target_energy) {
        abort();
      }
    }
  }
};

/// @brief Main function to generate a random map and solve the Traveling
/// Salesman Problem (TSP) using the Gurobi optimizer
/// @param argc Number of command-line arguments.
/// @param argv List of command-line arguments.
/// @details Generates a random map with the specified number of nodes and
/// random seed, and solves it using the Gurobi optimizer.
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

  std::cout << "Generating a TSP QUBO model" << std::endl;
  qbpp::tsp::TSPQuadModel tsp_quad_model(tsp_map, fix_first);

  std::cout << "Variables = " << tsp_quad_model.var_count()
            << " Linear Terms = " << tsp_quad_model.term_count(1)
            << " Quadratic Terms = " << tsp_quad_model.term_count(2)
            << std::endl;

  qbpp_grb::QuadModel grb_model(tsp_quad_model);
  grb_model.set_time_limit(time_limit);
  GRB_Callback grb_callback(tsp_quad_model, 0);
  grb_model.set(grb_callback);

  std::cout << "Solving the TSP" << std::endl;
  auto sol = grb_model.optimize();

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
