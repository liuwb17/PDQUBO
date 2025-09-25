/// @file qbpp_tsp.hpp
/// @author Koji Nakano
/// @brief Generates a QUBO Expression for the Traveling Salesman Problem (TSP)
/// using QUBO++ library.
/// @version 2025-03-04

#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "qbpp.hpp"
#include "qbpp_misc.hpp"

namespace qbpp {

/// @brief Namespace for the Traveling Salesman Problem (TSP) using QUBO++
namespace tsp {

constexpr uint32_t uint32_limit = std::numeric_limits<uint32_t>::max();

/// @brief Class to generates a random map for the TSP with n nodes
/// @details The map is a grid of size grid_size x grid_size
/// @note The nodes are randomly placed on the grid so that nodes are not too
/// close to each other
class TSPMap {
  /// @brief Size of the grid. grid_size x grid_size coordinates are used for
  /// the map.
  const uint32_t grid_size;
  /// @brief List of nodes with coordinate (x, y)
  std::vector<std::pair<int32_t, int32_t>> nodes;

public:
  /// @brief Constructor: Creates an empty map.
  /// @param grid_size Size of the grid. grid_size x grid_size coordinates are
  /// used for the map.
  TSPMap(uint32_t grid_size = 100) : grid_size(grid_size) {};

  /// @brief Generate a random map with n nodes
  /// @param n Number of nodes
  void gen_random_map(uint32_t n);

  /// @brief Add a node to the map
  /// @param x x-coordinate of the node
  /// @param y y-coordinate of the node
  void add_node(uint32_t x, uint32_t y) { nodes.push_back({x, y}); }

  /// @brief Compute the Euclidean distance between two nodes
  /// @param p1 First node
  /// @param p2 Second node
  /// @return Euclidean distance between the two nodes
  /// @note The distance is rounded to the nearest integer
  uint32_t dist(const std::pair<int32_t, int32_t> &p1,
                const std::pair<int32_t, int32_t> &p2) const {
    return static_cast<uint32_t>(std::round(
        std::sqrt((p1.first - p2.first) * (p1.first - p2.first) +
                  (p1.second - p2.second) * (p1.second - p2.second))));
  }

  /// @brief Compute the Euclidean distance between two nodes
  /// @param i Index of the first node
  /// @param j Index of the second node
  /// @return Euclidean distance between the two nodes
  uint32_t dist(uint32_t i, uint32_t j) const {
    return dist(nodes[i], nodes[j]);
  }

  /// @brief Compute the minimum distance between a new node and all other
  /// nodes
  /// @param x x-coordinate of the node
  /// @param y y-coordinate of the node
  /// @return Minimum distance between the node and all other nodes
  uint32_t min_dist(uint32_t x, uint32_t y) const {
    uint32_t min_dist = grid_size * 2;
    for (const auto &[px, py] : nodes) {
      if (dist({x, y}, {px, py}) < min_dist)
        min_dist = dist({x, y}, {px, py});
    }
    return min_dist;
  }

  /// @brief Gets the number of nodes in the map
  /// @return Number of nodes in the map
  uint32_t node_count() const { return static_cast<uint32_t>(nodes.size()); }

  /// @brief Gets the size of the grid
  /// @return Size of the grid
  uint32_t get_grid_size() const { return grid_size; }

  /// @brief Get the position of the node at index i
  /// @param index Index of the node
  /// @return Pair of coordinates (x, y) of the node
  std::pair<int32_t, int32_t> &operator[](uint32_t index) {
    return nodes[index];
  }
};

/// @brief Class to store the QUBO expression for the Traveling Salesman
/// Problem (TSP).
class TSPQuadModel : public qbpp::QuadModel {
  /// @brief true if the first node is fixed to node 0.
  const bool fix_first;
  /// @brief Variables for the TSP.
  const qbpp::Vector<qbpp::Vector<qbpp::Var>> x;

  /// @brief Generate a QUBO expression for the Traveling Salesman Problem
  /// (TSP).
  /// @return QUBO expression for the Traveling Salesman Problem (TSP).
  /// @note Helper function for the constructor.
  std::tuple<qbpp::Model, bool, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
  helper_func(const TSPMap &map, bool fix_first);

  TSPQuadModel(
      std::tuple<qbpp::QuadModel, bool, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
          tuple)
      : qbpp::QuadModel(std::get<0>(tuple)), fix_first(std::get<1>(tuple)),
        x(std::get<2>(tuple)) {}

public:
  /// @brief Generate a QUBO expression for the Traveling Salesman Problem
  /// (TSP) from a map.
  /// @param map Map of nodes with their coordinates
  /// @param fix_first true if the first node is fixed to node 0
  TSPQuadModel(const TSPMap &map, bool fix_first = false)
      : TSPQuadModel(helper_func(map, fix_first)) {}

  /// @brief Returns the number of nodes in the TSP.
  /// @return Number of nodes in the TSP.
  uint32_t node_count() const { return static_cast<uint32_t>(x.size()); }

  /// @brief Get the variable at (i, j).
  qbpp::Var get_var(uint32_t i, uint32_t j) const { return x[i][j]; }

  /// @brief Returns true if the first node is fixed to node 0.
  bool get_fix_first() const { return fix_first; }
};

/// @brief Class to store a Tour of the TSP
class TSPSol {
  /// @brief QUBO expression for the Traveling Salesman Problem (TSP).
  const TSPQuadModel tsp_quad_model;
  /// @brief Solution of the QUBO model.
  const Sol sol;
  /// @brief A vector of nodes representing the tour.
  const std::vector<uint32_t> tour;

  /// @brief helper function to generate a tour from the solution.
  /// @param tsp_quad_model TSP QUBO expression
  /// @param sol Solution of the QUBO model
  /// @return A vector of nodes representing the tour.
  static std::vector<uint32_t> gen_tour(const TSPQuadModel &tsp_quad_model,
                                        const Sol &sol);

public:
  /// @brief Generates a solution of the Traveling Salesman Problem (TSP) from
  /// a QUBO model and its solution.
  /// @param tsp_quad_model The TSP QUBO expression
  /// @param sol The Solution of the QUBO model
  TSPSol(const TSPQuadModel &tsp_quad_model, const qbpp::Sol &sol)
      : tsp_quad_model(tsp_quad_model), sol(sol),
        tour(gen_tour(tsp_quad_model, sol)) {}

  /// @brief Get the node at index in the tour.
  /// @param index Index of the node.
  /// @return Node at index in the tour.
  uint32_t operator[](uint32_t index) const { return tour[index]; }

  /// @brief Get the number of nodes in the tour.
  uint32_t node_count() const { return tsp_quad_model.node_count(); }

  /// @brief Print the tour.
  void print() const {
    std::cout << sol.energy() << " :";
    for (const auto &i : tour)
      if (i != uint32_limit)
        std::cout << " " << i;
      else
        std::cout << " -";
    std::cout << std::endl;
  }

  void print_matrix() const {
    for (uint32_t i = 0; i < tsp_quad_model.node_count(); i++) {
      for (uint32_t j = 0; j < tsp_quad_model.node_count(); j++) {
        if (i == 0 && j == 0)
          std::cout << "1";
        else if (i == 0 || j == 0)
          std::cout << "0";
        else {
          std::cout << sol.get(tsp_quad_model.get_var(i, j));
        }
      }
      std::cout << std::endl;
    }
  }
};

/// @brief Class to store the QUBO expression for the Traveling Salesman
/// Problem (TSP).
class TSPModel : public qbpp::QuadModel {
  /// @brief true if the first node is fixed to node 0.
  const bool fix_first;
  /// @brief Variables for the TSP.
  const qbpp::Vector<qbpp::Vector<qbpp::Var>> x;

  /// @brief Generate a QUBO expression for the Traveling Salesman Problem
  /// (TSP).
  /// @return QUBO expression for the Traveling Salesman Problem (TSP).
  /// @note Helper function for the constructor.
  std::tuple<qbpp::QuadModel, bool, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
  helper_func(const TSPMap &map, bool fix_first);

  TSPModel(std::tuple<qbpp::QuadModel, bool,
                      qbpp::Vector<qbpp::Vector<qbpp::Var>>> &&tuple)
      : qbpp::QuadModel(std::move(std::get<0>(tuple))),
        fix_first(std::get<1>(tuple)), x(std::move(std::get<2>(tuple))) {}

public:
  /// @brief Generate a QUBO expression for the Traveling Salesman Problem
  /// (TSP) from a map.
  /// @param map Map of nodes with their coordinates
  /// @param fix_first true if the first node is fixed to node 0
  TSPModel(const TSPMap &map, bool fix_first = false)
      : TSPModel(helper_func(map, fix_first)) {}

  /// @brief Returns the number of nodes in the TSP.
  /// @return Number of nodes in the TSP.
  uint32_t node_count() const { return static_cast<uint32_t>(x.size()); }

  /// @brief Get the variable at (i, j).
  qbpp::Var get_var(uint32_t i, uint32_t j) const { return x[i][j]; }

  /// @brief Returns true if the first node is fixed to node 0.
  bool get_fix_first() const { return fix_first; }
};

/// @brief Class to draw a simple undirected graph.
/// @note Nodes must be numbered from 0 to size-1.
/// @note Graphvis must be installed to use this class.
class DrawSimpleGraph {
  /// @brief List of nodes with coordinate (x, y)
  /// @note The graph is undirected
  std::vector<std::tuple<int, int, std::string>> nodes;
  /// @brief List of edges (node1, node2)
  std::vector<std::tuple<int, int>> edges;

public:
  /// @brief Default constructor to create an empty graph
  DrawSimpleGraph() = default;

  ///@brief Add a node to the graph
  ///@param x x-coordinate of the node
  ///@param y y-coordinate of the node
  ///@param label Label of the node
  void add_node(int x, int y, const std::string &label = "") {
    nodes.push_back(std::make_tuple(x, y, label));
  }
  /// @brief Add a node to the graph
  /// @param node Pair of coordinates (x, y) of the node
  /// @param label Label of the node
  void add_node(std::pair<int, int> node, const std::string &label = "") {
    add_node(node.first, node.second, label);
  }

  /// @brief Add an edge to the graph
  /// @param node1 Index of the first node
  /// @param node2 Index of the second node
  /// @note if node1 > node2, the nodes are swapped
  void add_edge(unsigned int node1, unsigned int node2) {
    edges.push_back(std::make_pair(node1, node2));
  }
  /// @brief Get the number of nodes in the graph
  uint32_t node_count() const { return static_cast<uint32_t>(nodes.size()); }

  /// @brief Get the number of edges in the graph
  uint32_t edge_count() const { return static_cast<uint32_t>(edges.size()); }

  /// @brief Draw the graph in a file
  /// @param filename Name of the file to save the graph
  /// @note The file format is determined by the extension of the filename
  /// @note The graph is drawn using the neato program from the Graphviz suite
  void draw(std::string filename) {
    std::ostringstream dot_stream;
    dot_stream << "graph G {\n"
               << "node [shape=circle, fixedsize=true, width=5, fontsize=200, "
                  "penwidth=10];\n"
               << "edge [penwidth=10];\n";
    int index = 0;
    for (auto [x, y, s] : nodes) {
      dot_stream << index << " [label = \"" << index << "\", pos = \"" << x
                 << "," << y << "!\"";
      if (s != "")
        dot_stream << " " << s;
      dot_stream << "];\n";
      ++index;
    }
    for (auto [node1, node2] : edges) {
      dot_stream << node1 << " -- " << node2 << "\n";
    }
    dot_stream << "}\n";
    std::string command = "neato -T" +
                          filename.substr(filename.rfind('.') + 1) + " -o " +
                          filename;

    std::unique_ptr<FILE, qbpp::misc::PcloseDeleter> pipe(
        popen(command.c_str(), "w"));

    if (!pipe) {
      throw std::runtime_error(THROW_MESSAGE("popen() failed!"));
    }
    fprintf(pipe.get(), "%s", dot_stream.str().c_str());
  }
};

//==============================
// TSPMap member functions
//==============================

inline void TSPMap::gen_random_map(uint32_t n) {
  nodes.reserve(n);
  uint32_t x, y;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t counter = 0;
    uint32_t max_dist = 1;
    // Terminate if dist>=max_dist is satisfied 10 times.
    while (counter < 10) {
      x = qbpp::misc::RandomGenerator::gen(grid_size);
      y = qbpp::misc::RandomGenerator::gen(grid_size);
      uint32_t dist = min_dist(x, y);
      if (dist >= max_dist) {
        max_dist = dist;
        counter++;
      }
    }
    add_node(x, y);
  }
}

//===============================
// TSPQuadModel member functions
//===============================

inline std::tuple<qbpp::Model, bool, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
TSPQuadModel::helper_func(const TSPMap &tsp_map, bool fix_first) {
  auto node_count = tsp_map.node_count();
  auto x = qbpp::var("x", node_count, node_count);
  std::cout << "Generating QUBO expression for permutation." << std::endl;
  auto permutation_expr = qbpp::sum(
      (qbpp::vector_sum(x) == 1) + (qbpp::vector_sum(qbpp::transpose(x)) == 1));
  std::cout << "Generating QUBO expressions for tour distances." << std::endl;

  qbpp::Vector<qbpp::Expr> exprs(node_count);
  tbb::parallel_for(decltype(node_count)(0), node_count, [&](int i) {
    uint32_t next_i = (i + 1) % node_count;
    auto &expr = exprs[i];

    for (uint32_t j = 0; j < node_count; j++) {
      for (uint32_t k = 0; k < node_count; k++) {
        if (j == k)
          continue;
        expr += tsp_map.dist(j, k) * x[i][j] * x[next_i][k];
      }
    }
  });

  auto tsp_expr = qbpp::sum(exprs) + permutation_expr * tsp_map.get_grid_size();

  if (fix_first) {
    std::cout << "Fixing the first visiting node." << std::endl;
    qbpp::MapList fix0 = {{x[0][0], 1}};
    for (uint32_t i = 1; i < node_count; i++) {
      fix0.push_back({x[0][i], 0});
      fix0.push_back({x[i][0], 0});
    }
    tsp_expr.replace(fix0);
  }

  std::cout << "Simplifying the QUBO expression." << std::endl;

  tsp_expr.simplify_as_binary();

  return {tsp_expr, fix_first, x};
}

//=========================
// TSPSol member functions
//=========================

inline std::vector<uint32_t>
TSPSol::gen_tour(const TSPQuadModel &tsp_quad_model, const Sol &sol) {
  std::vector<uint32_t> tour;
  for (uint32_t i = 0; i < tsp_quad_model.node_count(); i++) {
    if (tsp_quad_model.get_fix_first() && i == 0) {
      tour.push_back(0);
      continue;
    }
    uint32_t count = 0;
    uint32_t node;
    for (uint32_t j = (tsp_quad_model.get_fix_first() ? 1 : 0);
         j < tsp_quad_model.node_count(); j++) {
      if (sol.get(tsp_quad_model.get_var(i, j)) == 1) {
        node = j;
        count++;
      }
    }
    if (count != 1)
      node = uint32_limit;
    tour.push_back(node);
  }
  return tour;
}

} // namespace tsp
} // namespace qbpp
