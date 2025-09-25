
/// @file qbpp_nqueen.hpp
/// @brief Generates QUBO expression for the N-Queens problem using the QUBO++
/// library
/// @details This file provides a class to generate a QuadModel object for the
/// N-Queens problem using the QUBO++ library
/// @author Koji Nakano
/// @version 2025-01-12

#ifndef QBPP_NQUEEN_HPP
#define QBPP_NQUEEN_HPP

#include "qbpp.hpp"

namespace qbpp {
/// @brief Namespace for the N-Queens problem
namespace nqueen {

/// @brief Class to generate a QUBO model for the N-Queens problem
/// @details This class is a derived class of qbpp::QuadModel with the
/// 2-dimensional vector of Var objects representing the chessboard.
class NQueenQuadModel : public qbpp::QuadModel {
 public:
  enum class Mode { EXPAND, FAST, PARALLEL };

 private:
  /// @brief Dimension of the chessboard
  const int dim_;
  /// @brief 2-dimensional vector of qbpp::Var representing the chessboard
  /// @details The Var object X_[i][j] is 1 if a queen is placed at (i, j) and 0
  const Vector<Vector<qbpp::Var>> X_;

  /// @brief Generates qbpp::Expr for the N-Queens problem
  /// @return qbpp::Expr for the N-Queens problem
  /// @details The expression is created using the one-hot and zero-one-hot
  /// constraints. If all constraints are satisfied, board variables store
  /// correct placement of queens and the energy value is 0.
  static qbpp::QuadModel expand_mode(int dim,
                                     const Vector<Vector<qbpp::Var>> &X) {
    // Create arrays of expressions to store the sums of the diagonals.

    auto a = qbpp::expr(2 * dim - 3);
    auto b = qbpp::expr(2 * dim - 3);
    for (int i = 0; i < 2 * dim - 3; i++) {
      int k = i + 1;
      for (int j = 0; j < dim; j++) {
        if (k - j >= 0 && k - j < dim) {
          a[i] += X[j][k - j];
        }
      }
      int l = dim - i - 1;
      for (int j = 0; j < dim; j++) {
        if (l + j >= 0 && l + j < dim) {
          b[i] += X[j][l + j];
        }
      }
    }

    auto expr = qbpp::sum(qbpp::vector_sum(X) == 1);
    expr += qbpp::sum(qbpp::vector_sum(qbpp::transpose(X)) == 1);
    expr += qbpp::sum(0 <= a <= 1);
    expr += qbpp::sum(0 <= b <= 1);
    expr.simplify_as_binary();

    return qbpp::QuadModel(expr);
  }

  /// @brief Generates the QUBO expression for the N-Queens problem
  /// @return QUBO expression for the N-Queens problem
  /// @details The expression is created by setting the penalty for placing
  /// two queens in the same row, column, and diagonal. If the placement of
  /// queens is correct, the value of the expression will be 0.
  static qbpp::QuadModel fast_mode(int dim,
                                   const Vector<Vector<qbpp::Var>> &X) {
    qbpp::Expr expr = dim;
    for (int x = 0; x < dim; ++x) {
      for (int y = 0; y < dim; ++y) {
        // Reward for placing a queen
        expr -= X[x][y];
        for (int i = x + 1; i < dim; ++i) {
          // Penalty for placing two queens in the same row
          expr += X[x][y] * X[i][y];
        }
        for (int j = y + 1; j < dim; ++j) {
          // Penalty for placing two queens in the same column
          expr += X[x][y] * X[x][j];
        }
        for (int d = 1; d < dim - x; ++d) {
          if (y - d >= 0) {
            // Penalty for placing two queens on the same anti-diagonal
            expr += X[x][y] * X[x + d][y - d];
          }
          if (y + d < dim) {
            // Penalty for placing two queens on the same diagonal
            expr += X[x][y] * X[x + d][y + d];
          }
        }
      }
    }
    return qbpp::QuadModel(expr.simplify_as_binary());
  }

  /// @brief Generates the QUBO expression for the N-Queens problem using TBB
  /// in parallel.
  static qbpp::QuadModel parallel_mode(int dim,
                                       const Vector<Vector<qbpp::Var>> &X) {
    qbpp::Vector<qbpp::Expr> exprs(dim);
    tbb::parallel_for(0, dim, [&](int x) {
      for (int y = 0; y < dim; ++y) {
        // Reward for placing a queen
        exprs[x] -= X[x][y];
        for (int i = x + 1; i < dim; ++i) {
          // Penalty for placing two queens in the same row
          exprs[x] += X[x][y] * X[i][y];
        }
        for (int j = y + 1; j < dim; ++j) {
          // Penalty for placing two queens in the same column
          exprs[x] += X[x][y] * X[x][j];
        }
        for (int d = 1; d < dim - x; ++d) {
          if (y - d >= 0) {
            // Penalty for placing two queens on the same anti-diagonal
            exprs[x] += X[x][y] * X[x + d][y - d];
          }
          if (y + d < dim) {
            // Penalty for placing two queens on the same diagonal
            exprs[x] += X[x][y] * X[x + d][y + d];
          }
        }
      }
    });

    return qbpp::QuadModel(simplify_as_binary(qbpp::sum(exprs) + dim));
  }

  /// @brief Helper function to compute initial values for member variables.
  /// @param dim The dimension of the chessboard.
  /// @param mode The mode to generate the QUBO expression.
  /// @return A tuple containing the QUBO model, dimension, and variables.
  std::tuple<qbpp::QuadModel, int, Vector<Vector<qbpp::Var>>> helper_func(
      int dim, Mode mode) {
    Vector<Vector<qbpp::Var>> X(qbpp::var("X", dim, dim));
    switch (mode) {
      case Mode::EXPAND:
        return {expand_mode(dim, X), dim, X};
      case Mode::FAST:
        return {fast_mode(dim, X), dim, X};
      case Mode::PARALLEL:
        return {parallel_mode(dim, X), dim, X};
      default:
        throw std::runtime_error("Invalid mode");
    }
  }

  /// @brief Constructor to initialize member variables.
  /// @param tuple A tuple containing values to initialize the member
  /// variables.
  /// @note Since the arguments are the return values of the helper function,
  /// the rvalue reference is used.
  NQueenQuadModel(
      std::tuple<qbpp::QuadModel, int, Vector<Vector<qbpp::Var>>> &&tuple)
      : qbpp::QuadModel(std::get<0>(tuple)),
        dim_(std::get<1>(tuple)),
        X_(std::get<2>(tuple)) {}

 public:
  /// @brief Constructor: Creates an N-Queens QUBO expression
  /// @param dim Dimension of the chessboard
  /// @param mode Mode to generate the QUBO expression
  NQueenQuadModel(int dim, Mode mode)
      : NQueenQuadModel(helper_func(dim, mode)) {}

  /// @brief Gets the variable at (i, j)
  /// @param i Row index
  /// @param j Column index
  /// @return Var object at (i, j)
  qbpp::Var get_var(int i, int j) const { return X_[i][j]; }
};
}  // namespace nqueen
}  // namespace qbpp

#endif  // QBPP_NQUEEN_HPP