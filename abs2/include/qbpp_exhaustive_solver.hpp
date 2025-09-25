/// @file qbpp_exhaustive_solver.hpp
/// @author Koji Nakano
/// @brief Exhaustive QUBO Solver for solving QUBO problems.
/// @details The ExhaustiveSolver class provides a straightforward QUBO solver
/// that evaluates all possible solutions. This solver is primarily intended
/// for testing the correctness of QUBO expressions.
/// For more details on the algorithm, please refer to the following paper:
/// Masaki Tao et al., "A Work-Time Optimal Parallel Exhaustive Search
/// Algorithm for the QUBO and the Ising Model, with GPU Implementation," IPDPS
/// Workshops 2020: 557-566. https://doi.org/10.1109/IPDPSW50202.2020.00098
/// @version 2025-01-19

#ifndef QBPP_EXHAUSTIVE_SOLVER_HPP
#define QBPP_EXHAUSTIVE_SOLVER_HPP

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <boost/circular_buffer.hpp>
#include <random>
#include <set>

#include "qbpp.hpp"

namespace qbpp {

namespace exhaustive_solver {


class SolDelta;
class Sol;
class SearchAlgorithm;
class ExhaustiveSolver;

class Sol : public qbpp::Sol {
  bool is_all_solutions_ = false;

  bool is_optimal_solutions_ = false;

  std::vector<qbpp::Sol> all_solutions_;

public:
  explicit Sol(const QuadModel &quad_model) : qbpp::Sol(quad_model) {};

  Sol(const Sol &) = default;

  Sol(Sol &&) = default;

  Sol &operator=(const Sol &) = default;

  Sol &operator=(Sol &&) = default;

  std::vector<qbpp::Sol> &get_all_solutions() { return all_solutions_; }

  qbpp::Sol get_sol() const { return all_solutions_.front(); }

  void all_solution_mode() {
    is_optimal_solutions_ = false;
    is_all_solutions_ = true;
  }

  void optimal_solution_mode() {
    is_optimal_solutions_ = true;
    is_all_solutions_ = false;
  }

  void set_all_solutions(std::vector<qbpp::Sol> &&all_solutions) {
    all_solutions_ = all_solutions;
    qbpp::Sol::operator=(all_solutions_.front());
  }

  std::string str() const {
    if (is_all_solutions_ || is_optimal_solutions_) {
      std::ostringstream oss;
      uint32_t count = 0;
      for (const auto &sol : all_solutions_) {
        oss << count++ << ":" << sol;
        if (count < all_solutions_.size())
          oss << std::endl;
      }
      return oss.str();
    }
    return qbpp::str(static_cast<qbpp::Sol>(*this));
  }

  std::vector<qbpp::Sol>::const_iterator begin() const {
    return all_solutions_.begin();
  }

  std::vector<qbpp::Sol>::const_iterator end() const {
    return all_solutions_.end();
  }

  size_t size() const { return all_solutions_.size(); }

  const qbpp::Sol &operator[](size_t i) const { return all_solutions_[i]; }

  qbpp::Sol &operator[](size_t i) { return all_solutions_[i]; }
};

inline std::ostream &operator<<(std::ostream &os, const Sol &sol) {
  os << sol.str();
  return os;
}

class ExhaustiveSolver {
protected:
  const QuadModel quad_model_;

  bool enable_default_callback_ = false;

public:

  ExhaustiveSolver(const QuadModel &quad_model) : quad_model_(quad_model) {}

  mutable std::mutex callback_mutex_;

  virtual ~ExhaustiveSolver() = default;

  virtual void callback(const SolHolder &sol_holder) const {
    static std::optional<energy_t> prev_energy = std::nullopt;
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (enable_default_callback_) {
      if (!prev_energy.has_value() ||
          sol_holder.energy() < prev_energy.value()) {
        prev_energy = sol_holder.energy();
        std::cout << "TTS = " << std::fixed << std::setprecision(3)
                  << std::setfill('0') << sol_holder.get_tts()
                  << "s Energy = " << sol_holder.energy() << std::endl;
      }
    }
  }

  vindex_t var_count() const { return quad_model_.var_count(); }

  qbpp::Sol search();

  Sol search_optimal_solutions();

  Sol search_all_solutions();

  void enable_default_callback(bool enable = true) {
    enable_default_callback_ = enable;
  }


  const QuadModel &get_quad_model() const { return quad_model_; }
};

class SearchAlgorithm {
  const ExhaustiveSolver &exhaustive_solver_;

  const QuadModel &quad_model_;

  const std::vector<vindex_t> var_order_;

  qbpp::SolHolder sol_holder_;

  bool is_all_solutions_ = false;

  bool is_optimal_solutions_ = false;

  std::vector<qbpp::Sol> all_solutions_;

  mutable std::mutex all_solutions_mutex_;

  std::vector<SolDelta> sol_deltas;

  std::vector<vindex_t> init_var_order(const QuadModel &quad_model) {
    std::vector<std::pair<vindex_t, vindex_t>> degree_var;
    degree_var.resize(quad_model.var_count());
    for (vindex_t i = 0; i < quad_model.var_count(); ++i) {
      degree_var[i] = std::make_pair(quad_model.degree(i), i);
    }
    std::sort(degree_var.begin(), degree_var.end(), std::greater<>());
    std::vector<vindex_t> var_order;
    var_order.resize(quad_model.var_count());
    for (vindex_t i = 0; i < quad_model.var_count(); ++i) {
      var_order[i] = degree_var[i].second;
    }
    return var_order;
  }

public:
  explicit SearchAlgorithm(const ExhaustiveSolver &exhaustive_solver)
      : exhaustive_solver_(exhaustive_solver),
        quad_model_(exhaustive_solver_.get_quad_model()),
        var_order_(init_var_order(quad_model_)), sol_holder_(quad_model_) {}

  const QuadModel &get_quad_model() const { return quad_model_; }

  const std::vector<vindex_t> &get_var_order() const { return var_order_; }

  vindex_t var_count() const { return quad_model_.var_count(); }

  std::vector<qbpp::Sol> &get_all_solutions() { return all_solutions_; }

  qbpp::SolHolder &get_sol_holder() { return sol_holder_; }

  void register_new_sol(const qbpp::Sol &sol) {
    if (!is_all_solutions_ && sol_holder_.energy() < sol.energy())
      return;
    std::lock_guard<std::mutex> lock(all_solutions_mutex_);
    if (is_all_solutions_) {
      all_solutions_.push_back(sol);
    } else if (is_optimal_solutions_) {
      if (sol_holder_.energy() == sol.energy()) {
        all_solutions_.push_back(sol);
      } else if (sol_holder_.energy() > sol.energy()) {
        all_solutions_.clear();
        all_solutions_.push_back(sol);
      }
    }
    if (sol_holder_.set_if_better(sol)) {
      exhaustive_solver_.callback(sol_holder_);
    }
  }

  void search();

  void search_optimal_solutions();

  void search_all_solutions();

  void gen_sol_deltas(SolDelta &sol_delta, vindex_t index);
};

class SolDelta : public qbpp::Sol {
  SearchAlgorithm &search_algorithm_;

  const std::vector<vindex_t> &var_order_ = search_algorithm_.get_var_order();

  std::vector<energy_t> delta_;

public:
  explicit SolDelta(SearchAlgorithm &search_algorithm)
      : Sol(search_algorithm.get_quad_model()),
        search_algorithm_(search_algorithm) {
    energy_ = quad_model_.constant();
    delta_.resize(quad_model_.var_count());
    for (vindex_t i = 0; i < quad_model_.var_count(); ++i) {
      delta_[i] = quad_model_.linear(i);
    }
  }

  SolDelta(const SolDelta &) = default;

  vindex_t var_count() const { return quad_model_.var_count(); }

  void flip(vindex_t index) override {
    vindex_t flip_index = var_order_[index];
    if (flip_index >= var_count()) {
      throw std::out_of_range(
          THROW_MESSAGE("Sol: flip_index (", flip_index, ") out of range"));
    }
    energy_ = energy() + delta_[flip_index];
    for (vindex_t j = 0; j < quad_model_.degree(flip_index); ++j) {
      auto [k, coeff] = quad_model_.quadratic(flip_index, j);
      delta_[k] += (2 * get(flip_index) - 1) * (2 * get(k) - 1) * coeff;
    }
    delta_[flip_index] = -delta_[flip_index];
    if (!energy_.has_value()) {
      energy_ = comp_energy();
    }

    bit_vector_.flip(flip_index);
  }

  void search(vindex_t index) {
    if (index >= 1)
      search(index - 1);
    flip(index);
    search_algorithm_.register_new_sol(*this);
    if (index >= 1)
      search(index - 1);
  }

  void search() {
    search_algorithm_.register_new_sol(*this);
    search(quad_model_.var_count() - 1);
  }
};


inline qbpp::Sol ExhaustiveSolver::search() {
  SearchAlgorithm search_algorithm(*this);
  search_algorithm.search();
  return search_algorithm.get_sol_holder().get_sol();
}

inline Sol ExhaustiveSolver::search_optimal_solutions() {
  SearchAlgorithm search_algorithm(*this);
  search_algorithm.search_optimal_solutions();
  Sol sol(quad_model_);
  sol.optimal_solution_mode();
  sol.set_all_solutions(std::move(search_algorithm.get_all_solutions()));
  return sol;
}

inline Sol ExhaustiveSolver::search_all_solutions() {
  SearchAlgorithm search_algorithm(*this);
  search_algorithm.search_all_solutions();
  Sol sol(quad_model_);
  sol.all_solution_mode();
  sol.set_all_solutions(std::move(search_algorithm.get_all_solutions()));
  return sol;
}


inline void SearchAlgorithm::gen_sol_deltas(SolDelta &sol_delta,
                                            vindex_t index) {
  if (var_count() == index)
    return;
  gen_sol_deltas(sol_delta, index + 1);
  sol_delta.flip(index);
  sol_deltas.push_back(sol_delta);
  gen_sol_deltas(sol_delta, index + 1);
}

inline void SearchAlgorithm::search() {
  const int parallel_param = 8;
  SolDelta sol_delta(*this);
  if (quad_model_.term_count() == 0) {
    register_new_sol(sol_delta);
    return;
  }
  if (var_count() <= 16) {
    register_new_sol(sol_delta);
    sol_delta.search(var_count() - 1);
    return;
  }
  sol_deltas.push_back(sol_delta);
  gen_sol_deltas(sol_delta, var_count() - parallel_param);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, sol_deltas.size()),
                    [&](const tbb::blocked_range<size_t> &range) {
                      for (size_t i = range.begin(); i < range.end(); ++i) {
                        register_new_sol(sol_deltas[i]);
                        sol_deltas[i].search(var_count() -
                                             (parallel_param + 1));
                      }
                    });
}

inline void SearchAlgorithm::search_optimal_solutions() {
  is_optimal_solutions_ = true;
  search();
  tbb::parallel_sort(all_solutions_.begin(), all_solutions_.end());
}

inline void SearchAlgorithm::search_all_solutions() {
  is_all_solutions_ = true;
  search();
  tbb::parallel_sort(all_solutions_.begin(), all_solutions_.end());
}

} 
} 

#endif 