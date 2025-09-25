/// @file graph_color_abs2.cpp
/// @author Koji Nakano
/// @brief Solves randomly generated Graph Coloring Problem using ABS2 QUBO
/// Solver.
/// @version 2024-10-16

#include <boost/program_options.hpp>

#include "qbpp.hpp"
#include "qbpp_abs2.hpp"
#include "qbpp_graph_color.hpp"
#include "qbpp_misc.hpp"

namespace po = boost::program_options;

/// @brief Main function to generate a random map and solve the Graph Coloring
/// Problem using the ABS2 QUBO Solver.
/// @details Generates a random map with the specified number of nodes and
/// random seed, and solves it using the ABS2 QUBO Solver.
int main(int argc, char **argv) {
  // clang-format off
  po::options_description desc(
      "Solving the randomly generated Graph Coloring Problem using QUBO++ Easy Solver");
  desc.add_options()("help,h", "produce help message")
     ("nodes,n", po::value<uint32_t>()->default_value(100),"set the number of nodes")
     ("proximity,p", po::value<uint32_t>()->default_value(20),"set the proximity of the nodes")
     ("Circle,C","Set nodes are placed on a circle")
     ("Delaunay,D","Use Delaunay triangulation to connect the nodes")
     ("colors,c", po::value<uint32_t>()->default_value(4),"set the number of colors to use")
     ("time,t", po::value<uint32_t>()->default_value(10), "set time limit in seconds")
     ("problem_seed,s", po::value<uint32_t>(), "set the random seed for the TSP map")
     ("output,o", po::value<std::string>(), "set the output file (png, svg, etc) to save the TSP solution");
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
  uint32_t node_count = vm["nodes"].as<uint32_t>();
  /// @brief The time limit for the solver
  uint32_t time_limit = vm["time"].as<uint32_t>();
  /// @brief The number of colors to use
  uint32_t color_count = vm["colors"].as<uint32_t>();

  // Set the random seed for deterministic behavior if seed is provided.
  if (vm.count("problem_seed")) {
    qbpp::misc::RandomGenerator::set_seed(vm["problem_seed"].as<uint32_t>());
  } else {
    qbpp::misc::RandomGenerator::rd_seed();
  }

  std::cout << "Generating random graph with " << node_count << " nodes"
            << std::endl;
  qbpp::graph_color::GraphColorMap graph_color_map;

  graph_color_map.gen_random_map(node_count, vm.count("Circle"));

  if (vm.count("Delaunay")) {
    graph_color_map.gen_delaunay_edges();
  } else {
    graph_color_map.gen_proximity_edges(vm["proximity"].as<uint32_t>());
  }
  qbpp::graph_color::GraphColorQuadModel model(graph_color_map, color_count);

  std::cout << "Variables = " << model.var_count()
            << " Linear Terms = " << model.term_count(1)
            << " Quadratic Terms = " << model.term_count(2) << std::endl;

  if (vm.count("easy_solver_seed")) {
    qbpp::misc::RandomGenerator::set_seed(
        vm["easy_solver_seed"].as<uint32_t>());
  } else {
    qbpp::misc::RandomGenerator::rd_seed();
  }

  qbpp_abs2::Solver solver;
  qbpp_abs2::Param param;
  param.set_time_limit(time_limit);
  param.set_target_energy(0);
  auto sol = solver(model, param);

  graph_color_map.set_color_histogram(model, sol);

  graph_color_map.print();

  if (vm.count("output") > 0) {
    std::cout << "Writing the solution to " << vm["output"].as<std::string>()
              << std::endl;
    graph_color_map.draw(vm["output"].as<std::string>());
  }
}
