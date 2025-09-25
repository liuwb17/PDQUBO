/// @file qbpp_color.hpp
/// @author Koji Nakano
/// @brief Generates a QUBO Expression for the Graph Coloring Problem
/// using QUBO++ library.
/// @version 2024-10-16

#include <boost/polygon/voronoi.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "qbpp.hpp"
#include "qbpp_misc.hpp"

namespace qbpp {

/// @brief Namespace for the Traveling Salesman Problem (TSP) using QUBO++
namespace graph_color {

constexpr uint32_t uint32_limit = std::numeric_limits<uint32_t>::max();

/// @brief Class to generates a random map for the TSP with n nodes
/// @details The map is a grid of size grid_size x grid_size
/// @note The nodes are randomly placed on the grid so that nodes are not too
/// close to each other
class GraphColorMap {
  /// @brief Size of the grid. grid_size x grid_size coordinates are used for
  /// the map.
  const uint32_t grid_size_;
  /// @brief List of nodes with coordinate (x, y)
  std::vector<std::pair<int32_t, int32_t>> nodes_;

  /// @brief List of edges with the distance between the nodes
  std::vector<std::pair<uint32_t, uint32_t>> edges_;

  std::vector<uint32_t> color_;

public:
  /// @brief Constructor: Creates an empty map.
  /// @param grid_size_ Size of the grid. grid_size x grid_size coordinates
  /// are used for the map.
  GraphColorMap(uint32_t grid_size = 100) : grid_size_(grid_size){};

  /// @brief Generate a random map with n nodes
  /// @param n Number of nodes
  void gen_random_map(uint32_t n);

  void gen_edge(uint32_t proximity);

  /// @brief Add a node to the map
  /// @param x x-coordinate of the node
  /// @param y y-coordinate of the node
  void add_node(uint32_t x, uint32_t y) { nodes_.push_back({x, y}); }

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
    return dist(nodes_[i], nodes_[j]);
  }

  /// @brief Compute the minimum distance between a new node and all other
  /// nodes
  /// @param x x-coordinate of the node
  /// @param y y-coordinate of the node
  /// @return Minimum distance between the node and all other nodes
  uint32_t min_dist(uint32_t x, uint32_t y) const {
    uint32_t min_dist = grid_size_ * 2;
    for (const auto &[px, py] : nodes_) {
      if (dist({x, y}, {px, py}) < min_dist)
        min_dist = dist({x, y}, {px, py});
    }
    return min_dist;
  }

  /// @brief Gets the number of nodes in the map
  /// @return Number of nodes in the map
  uint32_t node_count() const { return static_cast<uint32_t>(nodes_.size()); }

  /// @brief Gets the size of the grid
  /// @return Size of the grid
  uint32_t get_grid_size() const { return grid_size_; }

  const std::vector<std::pair<uint32_t, uint32_t>> get_edges() const {
    return edges_;
  }

  /// @brief Get the position of the node at index i
  /// @param index Index of the node
  /// @return Pair of coordinates (x, y) of the node
  std::pair<int32_t, int32_t> &operator[](uint32_t index) {
    return nodes_[index];
  }

  void set_color(const Vector<int32_t> &color) {
    color_.reserve(node_count());
    for (uint32_t i = 0; i < node_count(); i++) {
      color_.push_back(static_cast<uint32_t>(color[i] + 1));
    }
  }

  /// @brief Draw the graph in a file
  /// @param filename Name of the file to save the graph
  /// @note The file format is determined by the extension of the filename
  /// @note The graph is drawn using the neato program from the Graphviz suite
  void draw(const std::string &filename) {
    const std::vector<std::string> color_palette = {
        "#AAAAAA", // グレー エラーを意味．
        "#FF0000", // 赤
        "#00FF00", // 緑
        "#FFFF00", // 黄色
        "#00FFFF", // シアン
        "#FF00FF", // マゼンタ
        "#FFA500", // オレンジ
        "#800080", // 紫
        "#A52A2A", // 茶色
        "#008000", // ダークグリーン
        "#000080", // ネイビー
        "#FFD700", // ゴールド
        "#808080", // グレー
        "#FF1493", // ディープピンク
        "#00CED1", // ダークターコイズ
        "#ADFF2F"  // イエローグリーン
        "#0000FF", // 青
    };

    std::ostringstream dot_stream;
    dot_stream << "graph G {\n"
               << "node [shape=circle, fixedsize=true, width=3, fontsize=100, "
                  "penwidth=5];\n"
               << "edge [penwidth=5];\n";

    for (size_t i = 0; i < nodes_.size(); ++i) {
      dot_stream << i << " [label=\"" << i << "\", pos=\"" << nodes_[i].first
                 << "," << nodes_[i].second << "!\""
                 << ", fillcolor=\"" << color_palette[color_[i]] << "\""
                 << ", style=filled];\n";
    }

    for (const auto &[node1, node2] : edges_) {
      dot_stream << node1 << " -- " << node2 << ";\n";
    }

    dot_stream << "}\n";

    std::string command = "neato -T" +
                          filename.substr(filename.rfind('.') + 1) + " -o " +
                          filename;

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "w"),
                                                  pclose);

    if (!pipe) {
      throw std::runtime_error("popen() failed!");
    }

    std::string dot_content = dot_stream.str();
    fwrite(dot_content.c_str(), sizeof(char), dot_content.size(), pipe.get());
  }
};

/// @brief Class to store the QUBO expression for the Traveling Salesman
/// Problem (TSP).
class ColorQuadModel : public qbpp::QuadModel {
  /// @brief Variables for the TSP.
  const qbpp::Vector<qbpp::Vector<qbpp::Var>> x;

  std::pair<qbpp::Model, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
  helper_func(const GraphColorMap &map, uint32_t color_count);

  ColorQuadModel(
      std::pair<qbpp::QuadModel, qbpp::Vector<qbpp::Vector<qbpp::Var>>> pair)
      : qbpp::QuadModel(pair.first), x(pair.second) {}

public:
  /// @brief Generate a QUBO expression for the Traveling Salesman Problem
  /// (TSP) from a map.
  /// @param map Map of nodes with their coordinates
  /// @param fix_first true if the first node is fixed to node 0
  ColorQuadModel(const GraphColorMap &map, uint32_t color_count)
      : ColorQuadModel(helper_func(map, color_count)) {}

  /// @brief Returns the number of nodes in the TSP.
  /// @return Number of nodes in the TSP.
  uint32_t node_count() const { return static_cast<uint32_t>(x.size()); }

  const qbpp::Vector<qbpp::Vector<qbpp::Var>> &get_x() const { return x; }

  /// @brief Get the variable at (i, j).
  qbpp::Var get_var(uint32_t i, uint32_t j) const { return x[i][j]; }
};

//==============================
// GraphColorMap member functions
//==============================

void GraphColorMap::gen_random_map(uint32_t n) {
  nodes_.reserve(n);
  uint32_t x, y;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t counter = 0;
    uint32_t max_dist = 1;
    // Terminate if dist>=max_dist is satisfied 10 times.
    while (counter < 10) {
      x = qbpp::misc::RandomGenerator::gen(grid_size_);
      y = qbpp::misc::RandomGenerator::gen(grid_size_);
      uint32_t dist = min_dist(x, y);
      if (dist >= max_dist) {
        max_dist = dist;
        counter++;
      }
    }
    add_node(x, y);
  }
}

void GraphColorMap::gen_edge(uint32_t proximity) {
  for (size_t i = 0; i < nodes_.size(); i++) {
    for (size_t j = i + 1; j < nodes_.size(); j++) {
      if (dist(i, j) < proximity) {
        edges_.push_back({i, j});
      }
    }
  }
}

//===============================
// ColorQuadModel member functions
//===============================

std::pair<qbpp::Model, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
ColorQuadModel::helper_func(const GraphColorMap &graph_map,
                            uint32_t color_count) {
  auto x = qbpp::var("x", graph_map.node_count(), color_count);

  auto f = qbpp::sum(qbpp::sum(x) == 1);

  for (auto [i, j] : graph_map.get_edges()) {
    f += qbpp::sum(x[i] * x[j]);
  }
  return {simplify_as_binary(f), x};
}

//=========================
// ColorSol member functions
//=========================

} // namespace graph_color
} // namespace qbpp
