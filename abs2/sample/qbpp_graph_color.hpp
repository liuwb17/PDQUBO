/// @author Koji Nakano
/// @brief Generates a QUBO Expression for the Graph Coloring Problem
/// using QUBO++ library.
/// @version 2025-03-20

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

/// @brief Namespace for the Graph Node Coloring using QUBO++
namespace graph_color {

class GraphColorMap;
class GraphColorQuadModel;

/// @brief Class to store a graph with node coloring and related information
/// @details This class includes nodes with coordinates, edges between the
/// nodes, colors of the nodes, and the histogram of the colors.
class GraphColorMap {
  /// @brief Size of the grid. grid_size x grid_size coordinates are used for
  /// the map.
  const uint32_t grid_size_;

  /// @brief List of nodes with coordinate (x, y)
  /// @details nodes_[i] = (x, y) is the coordinate of the i-th node.
  std::vector<std::pair<int32_t, int32_t>> nodes_;

  /// @brief List of edges with the distance between the nodes
  /// @details edges_[i] = (j, k) means that the j-th and k-th nodes are
  /// connected.
  /// @note j < k is always satisfied.
  std::vector<std::pair<uint32_t, uint32_t>> edges_;

  /// @brief List of colors of the nodes
  /// @details color_[i] is the color of the i-th node.
  //  @note -1 means that the color is not determined.
  qbpp::Vector<int32_t> color_;

  /// @brief Histogram of the colors
  /// @details color_hist_[i] is the number of nodes with color i.
  std::vector<uint32_t> color_hist_;

  /// @brief Number of failure nodes
  /// @details failure_ is the number of nodes that have no color.
  /// @note failure means that the resulting one-hot vector is not valid.
  uint32_t failure_ = 0;

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
  uint32_t dist(size_t i, size_t j) const { return dist(nodes_[i], nodes_[j]); }

  /// @brief Compute the minimum distance between a new node and all other
  /// nodes
  /// @param x x-coordinate of the node
  /// @param y y-coordinate of the node
  /// @return Minimum distance between the node and all other nodes
  uint32_t min_dist(uint32_t x, uint32_t y) const {
    uint32_t min_dist = grid_size_ * 2;
    for (const auto &[px, py] : nodes_) {
      if (dist({x, y}, {px, py}) < min_dist) min_dist = dist({x, y}, {px, py});
    }
    return min_dist;
  }

  /// @brief Get the position of the node at index i
  /// @param index Index of the node
  /// @return Pair of coordinates (x, y) of the node
  std::pair<int32_t, int32_t> &operator[](uint32_t index) {
    return nodes_[index];
  }

 public:
  /// @brief Constructor to Create an empty map.
  /// @param grid_size Size of the grid. grid_size x grid_size coordinates
  /// are used for the map.
  GraphColorMap(uint32_t grid_size = 100) : grid_size_(grid_size) {};

  /// @brief Generate a random map with n nodes
  /// @param n Number of nodes
  /// @param is_circle true if all nodes are placed in a circle
  void gen_random_map(uint32_t n, bool is_circle = false);

  /// @brief Create an edges between nodes that are close to each other
  /// @param proximity the maximum distance between two nodes to create an edge
  void gen_proximity_edges(uint32_t proximity);

  /// @brief Create edges between nodes using the Delaunay triangulation
  void gen_delaunay_edges();

  /// @brief Set the color histogram of the nodes
  /// @param model QUBO model for the graph coloring
  /// @param sol Solution of the QUBO model
  /// @details This function sets the color histogram from the QUBO model with
  /// the solution.
  void set_color_histogram(const GraphColorQuadModel &model,
                           const qbpp::Sol &sol);

  /// @brief Gets the number of nodes in the map
  /// @return Number of nodes in the map
  uint32_t node_count() const { return static_cast<uint32_t>(nodes_.size()); }

  /// @brief Gets the size of the grid
  /// @return Size of the grid
  uint32_t get_grid_size() const { return grid_size_; }

  const std::vector<std::pair<uint32_t, uint32_t>> get_edges() const {
    return edges_;
  }

  /// @brief Draw the graph in a file
  /// @param filename Name of the file to save the graph
  /// @param is_blank true if the graph is drawn without colors
  /// @note The file format is determined by the extension of the filename
  /// @note The graph is drawn using the neato program from the Graphviz suite
  void draw(const std::string &filename, bool is_blank = false);

  /// @brief Displays the histogram of the colors
  /// @note set_color_histogram must be called before calling this function.
  void print() {
    for (size_t i = 0; i < color_hist_.size(); i++) {
      std::cout << "Color " << i << " : " << color_hist_[i] << std::endl;
    }
    std::cout << "Failure : " << failure_ << std::endl;
  }
};

/// @brief Class to store the QUBO expression with variables for the Graph
/// Coloring Problem
/// @details This class includes is a derived class of qbpp::QuadModel with the
/// variables.
class GraphColorQuadModel : public qbpp::QuadModel {
  /// @brief Variables for the Graph Coloring Problem
  const qbpp::Vector<qbpp::Vector<qbpp::Var>> x_;

  /// @brief Helper function to generate the QUBO expression for the
  /// GraphColorMap object and the number of colors.
  std::pair<qbpp::Model, qbpp::Vector<qbpp::Vector<qbpp::Var>>> helper_func(
      const GraphColorMap &map, uint32_t color_count);

  /// @brief Delegated constructor for the GraphColorQuadModel
  /// @param pair Pair of the QUBO model and the variables
  /// @details This constructor is used to create a GraphColorQuadModel object
  /// and variables.
  GraphColorQuadModel(
      std::pair<qbpp::QuadModel, qbpp::Vector<qbpp::Vector<qbpp::Var>>> pair)
      : qbpp::QuadModel(pair.first), x_(pair.second) {}

 public:
  /// @brief Generate a QUBO expression for the Graph Coloring Problem.
  /// @param map Map of nodes with their coordinates
  /// @param color_count Number of colors to use for the coloring
  GraphColorQuadModel(const GraphColorMap &map, uint32_t color_count)
      : GraphColorQuadModel(helper_func(map, color_count)) {}

  /// @brief Returns the number of nodes in the TSP.
  /// @return Number of nodes in the TSP.
  uint32_t node_count() const { return static_cast<uint32_t>(x_.size()); }

  /// @brief Returns the reference of the variables for the Graph Coloring
  /// Problem.
  const qbpp::Vector<qbpp::Vector<qbpp::Var>> &get_x() const { return x_; }
};

//==============================
// GraphColorMap member functions
//==============================

inline void GraphColorMap::gen_random_map(uint32_t n, bool is_circle) {
  nodes_.reserve(n);
  uint32_t x, y;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t counter = 0;
    uint32_t max_dist = 1;
    // Terminate if dist>=max_dist is satisfied 10 times.
    while (counter < 10) {
      if (is_circle) {
        do {
          x = qbpp::misc::RandomGenerator::gen(grid_size_);
          y = qbpp::misc::RandomGenerator::gen(grid_size_);
        } while ((x - grid_size_ / 2) * (x - grid_size_ / 2) +
                     (y - grid_size_ / 2) * (y - grid_size_ / 2) >
                 grid_size_ * grid_size_ / 4);
      } else {
        x = qbpp::misc::RandomGenerator::gen(grid_size_);
        y = qbpp::misc::RandomGenerator::gen(grid_size_);
      }
      uint32_t dist = min_dist(x, y);
      if (dist >= max_dist) {
        max_dist = dist;
        counter++;
      }
    }
    add_node(x, y);
  }
}

inline void GraphColorMap::gen_proximity_edges(uint32_t proximity) {
  for (size_t i = 0; i < nodes_.size(); i++) {
    for (size_t j = i + 1; j < nodes_.size(); j++) {
      if (dist(i, j) < proximity) {
        edges_.push_back({i, j});
      }
    }
  }
}

inline void GraphColorMap::gen_delaunay_edges() {
  typedef boost::polygon::point_data<int> Point;
  std::vector<Point> points;
  for (const auto &[x, y] : nodes_) {
    points.push_back(Point(x, y));
  }
  boost::polygon::voronoi_diagram<double> vd;
  boost::polygon::construct_voronoi(points.begin(), points.end(), &vd);
  for (const auto &cell : vd.cells()) {
    size_t source_index = cell.source_index();
    const auto *edge = cell.incident_edge();
    do {
      size_t twin_source_index = edge->twin()->cell()->source_index();
      if (source_index < twin_source_index)
        edges_.push_back({static_cast<uint32_t>(source_index),
                          static_cast<uint32_t>(twin_source_index)});
      edge = edge->next();
    } while (edge != cell.incident_edge());
  }
}

inline void GraphColorMap::set_color_histogram(const GraphColorQuadModel &model,
                                               const qbpp::Sol &sol) {
  color_ = qbpp::onehot_to_int(sol.get(model.get_x()));
  for (auto c : color_) {
    if (c < 0) {
      ++failure_;
    } else {
      if (c >= static_cast<decltype(c)>(color_hist_.size())) {
        color_hist_.resize(static_cast<size_t>(c) + 1, 0);
      }
      color_hist_[static_cast<size_t>(c)]++;
    }
  }
}

inline void GraphColorMap::draw(const std::string &filename, bool is_blank) {
  const std::vector<std::string> color_palette = {
      "#FFFFFF",  // 白 エラーを意味．
      "#FF0000",  // 赤
      "#00FF00",  // 緑
      "#FFFF00",  // 黄色
      "#00FFFF",  // シアン
      "#FF00FF",  // マゼンタ
      "#FFA500",  // オレンジ
      "#800080",  // 紫
      "#A52A2A",  // 茶色
      "#87CEEB",  // ライトブルー
      "#FFD700",  // ゴールド
      "#808080",  // グレー
      "#FF1493",  // ディープピンク
      "#00CED1",  // ダークターコイズ
      "#ADFF2F",  // イエローグリーン
      "#ADD8E6",  // ライトスカイブルー
      "#008000",  // ダークグリーン
      "#F0E68C",  // カーキ
      "#7FFF00",  // チャートリューズ
      "#40E0D0",  // ターコイズ
      "#DDA0DD",  // プラム
      "#FF4500",  // オレンジレッド
      "#DA70D6",  // オーキッド
      "#F08080",  // ライトコーラル
      "#87CEFA",  // スカイブルー（もう一つの明るい青）
      "#FF6347",  // トマト
      "#FFE4B5",  // モカ
      "#BA55D3",  // ミディアムオーキッド
      "#3CB371",  // ミディアムシーグリーン
      "#4682B4",  // スチールブルー
      "#B0E0E6",  // パウダーブルー
      "#7B68EE"   // ミディアムスレートブルー
  };

  std::ostringstream dot_stream;
  dot_stream << "graph G {\n"
             << "node [shape=circle, fixedsize=true, width=3, fontsize=100, "
                "penwidth=8];\n"
             << "edge [penwidth=8];\n";

  for (size_t i = 0; i < nodes_.size(); ++i) {
    dot_stream << i << " [label=\"" << i << "\", pos=\"" << nodes_[i].first
               << "," << nodes_[i].second << "!\"" << ", fillcolor=\"";
    if (is_blank) {
      dot_stream << "#FFFFFF";
    } else if (static_cast<size_t>(color_[i] + 1) < color_palette.size()) {
      dot_stream << color_palette[static_cast<size_t>(color_[i]) + 1];
    } else {
      dot_stream << color_palette[0];
    }
    dot_stream << "\", style=filled];\n";
  }

  for (const auto &[node1, node2] : edges_) {
    if (!is_blank && (color_[node1] == color_[node2] || color_[node1] < 0 ||
                      color_[node2] < 0)) {
      dot_stream << node1 << " -- " << node2
                 << " [style=dashed,penwidth=16,color=\"red\"];\n";
    } else {
      dot_stream << node1 << " -- " << node2 << ";\n";
    }
  }

  dot_stream << "}\n";

  std::string command =
      "neato -T" + filename.substr(filename.rfind('.') + 1) + " -o " + filename;

  std::unique_ptr<FILE, qbpp::misc::PcloseDeleter> pipe(
      popen(command.c_str(), "w"));

  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }

  std::string dot_content = dot_stream.str();
  fwrite(dot_content.c_str(), sizeof(char), dot_content.size(), pipe.get());
}

//======================================
// GraphColorQuadModel member functions
//======================================

inline std::pair<qbpp::Model, qbpp::Vector<qbpp::Vector<qbpp::Var>>>
GraphColorQuadModel::helper_func(const GraphColorMap &graph_map,
                                 uint32_t color_count) {
  auto x = qbpp::var("x", graph_map.node_count(), color_count);

  auto f = qbpp::sum(qbpp::vector_sum(x) == 1);

  for (auto [i, j] : graph_map.get_edges()) {
    f += qbpp::sum(x[i] * x[j]);
  }
  return {simplify_as_binary(f), x};
}

}  // namespace graph_color
}  // namespace qbpp
