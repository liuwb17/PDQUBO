/// @file qbpp_easy_solver.hpp
/// @author Koji Nakano
/// @brief Easy QUBO Solver for solving QUBO problems.
/// @details EasySolver is a simple QUBO solver that utilizes a greedy algorithm
/// and the Positive Min algorithm with Tabu search. Despite its simplicity, the
/// algorithm is highly effective at escaping local minima. This solver is
/// designed primarily for testing the QUBO++ library, as well as for
/// experimentation, evaluation, and learning purposes. For more details on the
/// Positive Min algorithm, please refer to the following paper:
///   - Hiroshi Kagawa et al., "High-throughput FPGA implementation for
///   quadratic unconstrained binary optimization", Concurr. Comput. Pract. Exp.
///   35(14) (2023),
/// https://doi.org/10.1002/cpe.6565
///
/// @copyright 2025, Koji Nakano
/// @version 2025-04-30

#ifndef QBPP_EASY_SOLVER_HPP
#define QBPP_EASY_SOLVER_HPP

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <boost/circular_buffer.hpp>
#include <limits>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "qbpp.hpp"
#include "qbpp_misc.hpp"

extern "C" {
#include "xxhash.h"
}

namespace qbpp {


namespace easy_solver {

class BestSols;
class SolDelta;
class TabuSolDelta;
class PosMinSolDelta;
class EasySolver;

#ifdef USE_128BIT_HASH
using HashType = boost::multiprecision::uint128_t;
#else
using HashType = uint64_t;
#endif

HashType bit_vector_hash(const qbpp::impl::BitVector &bit_vector) {
#ifdef USE_128BIT_HASH
  return XXH3_128bits(bit_vector.get_bits_ptr(), bit_vector.size64() * 8);
#else
  return XXH3_64bits(bit_vector.get_bits_ptr(), bit_vector.size64() * 8);
#endif
}

struct SolHash {
  HashType operator()(const qbpp::Sol &sol) const noexcept {
    return static_cast<size_t>(bit_vector_hash(sol.bit_vector()));
  }
};

class BestSols {
  const size_t max_best_sol_count_;
  std::vector<qbpp::Sol> sol_vector_;
  std::unordered_set<qbpp::Sol, SolHash> sol_set_;
  energy_t worst_energy_ = std::numeric_limits<energy_t>::max();

public:
  explicit BestSols(size_t max_best_sol_count = 0)
      : max_best_sol_count_(max_best_sol_count) {}

  void insert_if_better(const qbpp::Sol &sol) {
    static std::mutex mutex_;
    if (max_best_sol_count_ == 0)
      return;
    if (worst_energy_ < sol.energy())
      return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (sol_set_.count(sol))
      return;
    if (sol_vector_.size() < max_best_sol_count_) {
      sol_vector_.push_back(sol);
      sol_set_.insert(sol);
    } else {
      if (sol.energy() >= sol_vector_.back().energy())
        return;
      sol_set_.erase(sol_vector_.back());
      sol_vector_.back() = sol;
      sol_set_.insert(sol);
    }

    for (size_t i = sol_vector_.size() - 1; i > 0; --i) {
      if (sol_vector_[i].energy() < sol_vector_[i - 1].energy()) {
        std::swap(sol_vector_[i], sol_vector_[i - 1]);
      } else {
        break;
      }
    }
    worst_energy_ = sol_vector_.back().energy();
  }

  size_t size() const { return sol_vector_.size(); }
  const std::vector<qbpp::Sol> &get() const { return sol_vector_; }
  const qbpp::Sol &get(size_t i) const { return sol_vector_[i]; }
};

class SolDelta : public Sol {
protected:
  const EasySolver &easy_solver_;

  std::vector<energy_t> delta_;

  misc::MinHeap<energy_t> neg_set_;

  const std::shared_ptr<qbpp::SolHolder> sol_holder_ptr_;

  const std::shared_ptr<BestSols> best_sols_ptr_;

  uint64_t flip_count_ = 0;

public:
  explicit SolDelta(const EasySolver &easy_solver);

  virtual ~SolDelta() = default;

  energy_t get_delta(vindex_t i) const { return delta_[i]; }

  vindex_t var_count() const;

  void flip(vindex_t index) override;

  std::optional<double> set_if_better(const std::string &solver_name);

  vindex_t neg_count() const { return neg_set_.heap_size(); }

  vindex_t neg_random() { return neg_set_.select_at_random(); }

  vindex_t neg_min() const { return neg_set_.get_first(); }

  virtual void
  before_delta_updated([[maybe_unused]] vindex_t update_delta_index) {}

  virtual void
  after_delta_updated([[maybe_unused]] vindex_t update_delta_index) {}

  void greedy();

  void random_flip(size_t iteration);

  void move_to(qbpp::Sol destination);

  uint64_t get_flip_count() const { return flip_count_; }

  const std::shared_ptr<BestSols> &best_sols_ptr() const {
    return best_sols_ptr_;
  }
};

class TabuSolDelta : public SolDelta {
protected:
  const vindex_t tabu_size_;

  misc::Tabu tabu_;

public:
  TabuSolDelta(const EasySolver &easy_solver, vindex_t tabu_size)
      : SolDelta(easy_solver), tabu_size_(tabu_size),
        tabu_(var_count(), tabu_size) {}

  virtual ~TabuSolDelta() = default;

  void flip(vindex_t index) override {
    SolDelta::flip(index);
    tabu_.insert(index);
  }

  bool tabu_has(vindex_t index) const { return tabu_.has(index); }

  vindex_t non_tabu_random() { return tabu_.non_tabu_random(); }
};

class PosMinSolDelta : public TabuSolDelta {
  misc::MinHeap<energy_t> pos_set_;

public:
  explicit PosMinSolDelta(const EasySolver &easy_solver);

  virtual ~PosMinSolDelta() = default;

  vindex_t pos_count() const { return pos_set_.heap_size(); }

  vindex_t pos_min() const { return pos_set_.get_first(); }

  void before_delta_updated(vindex_t k) override {
    if (delta_[k] > 0) {
      pos_set_.erase(k);
    }
  }

  void after_delta_updated(vindex_t k) override {
    if (delta_[k] > 0) {
      pos_set_.insert(k, delta_[k]);
    }
  }

  void search(size_t iteration);

  void print() const {
    for (vindex_t i = 0; i < var_count(); ++i) {
      std::cout << "(" << i << "," << delta_[i] << ")";
    }
    std::cout << std::endl;
    neg_set_.print("NEG:");
    pos_set_.print("POS:");
  }
};

class EasySolver {
protected:
  const qbpp::QuadModel quad_model_;

  std::optional<uint32_t> time_limit_;

  size_t thread_count_{std::thread::hardware_concurrency() < 4
                           ? 4
                           : std::thread::hardware_concurrency()};

  std::optional<energy_t> target_energy_;

  const bool is_internal_sol_holder_;

  std::shared_ptr<qbpp::SolHolder> sol_holder_ptr_;

  std::shared_ptr<BestSols> best_sols_ptr_;

  mutable std::mutex callback_mutex_;

  bool enable_default_callback_ = false;

  void single_search(size_t thread_id);

  double start_time_;

  std::mutex flip_count_mutex_;
  uint64_t flip_count_ = 0;

  mutable std::optional<energy_t> prev_energy_ = std::nullopt;

public:

  explicit EasySolver(const qbpp::QuadModel &quad_model,
                      std::shared_ptr<qbpp::SolHolder> sol_holder_ptr = nullptr)
      : quad_model_(quad_model),
        is_internal_sol_holder_(sol_holder_ptr == nullptr),
        sol_holder_ptr_(sol_holder_ptr) {}


  virtual ~EasySolver() = default;

  const qbpp::QuadModel &get_quad_model() const { return quad_model_; }

  const std::optional<uint32_t> get_time_limit() const { return time_limit_; }

  const std::optional<energy_t> get_target_energy() const {
    return target_energy_;
  }

  const std::shared_ptr<qbpp::SolHolder> sol_holder_ptr() const {
    return sol_holder_ptr_;
  }

  vindex_t var_count() const { return quad_model_.var_count(); }

  void set_time_limit(uint32_t limit) { time_limit_ = limit; }

  void set_target_energy(energy_t energy) { target_energy_ = energy; }

  void set_thread_count(unsigned int count) { thread_count_ = count; }

  size_t get_thread_count() { return thread_count_; }

  qbpp::Sol get_sol() const { return sol_holder_ptr_->get_sol(); }

  const qbpp::Sol &set_sol(const qbpp::Sol &sol) {
    return sol_holder_ptr_->set_sol(sol);
  }

  double get_tts() const { return sol_holder_ptr_->get_tts(); }

  uint64_t get_flip_count() const { return flip_count_; }

  void enable_default_callback() { enable_default_callback_ = true; }

  virtual void callback(const qbpp::Sol &sol, double tts) const {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (enable_default_callback_) {
      if (!prev_energy_.has_value() || sol.energy() < prev_energy_.value()) {
        std::cout << "TTS = " << std::fixed << std::setprecision(3)
                  << std::setfill('0') << tts << "s Energy = " << sol.energy()
                  << " thread = "
                  << tbb::this_task_arena::current_thread_index() << std::endl;
        prev_energy_ = sol.energy();
      }
    }
  }

  qbpp::Sol search(bool has_initial_sol = false);

  qbpp::Sol search(const qbpp::Sol &initial_sol) {
    sol_holder_ptr_->set_sol(initial_sol);
    return search(true);
  }

  void enable_best_sols(size_t max_best_sol_count) {
    if (max_best_sol_count == 0) {
      best_sols_ptr_ = nullptr;
      return;
    }
    best_sols_ptr_ = std::make_shared<BestSols>(max_best_sol_count);
  }

  std::shared_ptr<BestSols> &best_sols_ptr() { return best_sols_ptr_; }
  const std::shared_ptr<BestSols> &best_sols_ptr() const {
    return best_sols_ptr_;
  }

  BestSols &best_sols() {
    if (best_sols_ptr_ == nullptr) {
      throw std::runtime_error(
          "BestSols is not set. Please call enable_best_sols() first.");
    }
    return *best_sols_ptr_;
  }
  const BestSols &best_sols() const {
    if (best_sols_ptr_ == nullptr) {
      throw std::runtime_error(
          "BestSols is not set. Please call enable_best_sols() first.");
    }
    return *best_sols_ptr_;
  }

};


inline vindex_t SolDelta::var_count() const { return easy_solver_.var_count(); }

inline SolDelta::SolDelta(const EasySolver &easy_solver)
    : Sol(easy_solver.get_quad_model()), easy_solver_(easy_solver),
      delta_(var_count()), neg_set_(var_count()),
      sol_holder_ptr_(easy_solver.sol_holder_ptr()),
      best_sols_ptr_(easy_solver.best_sols_ptr()) {
  for (vindex_t i = 0; i < var_count(); ++i) {
    delta_[i] = quad_model_.linear(i);
  }
}

inline void SolDelta::flip(vindex_t index) {
  if (index >= var_count()) {
    throw std::out_of_range(
        THROW_MESSAGE("Flip index (", index, ") is out of range."));
  }

  for (vindex_t j = 0; j < quad_model_.degree(index); ++j) {
    auto [k, coeff] = quad_model_.quadratic(index, j);

    if (delta_[k] < 0) {
      neg_set_.erase(k);
    }
    before_delta_updated(k);

    delta_[k] +=
        static_cast<energy_t>((2 * get(index) - 1) * (2 * get(k) - 1) * coeff);

    if (delta_[k] < 0) {
      neg_set_.insert(k, delta_[k]);
    }
    after_delta_updated(k);
  }

  if (delta_[index] < 0) {
    neg_set_.erase(index);
  }
  before_delta_updated(index);

  delta_[index] = -delta_[index];

  if (delta_[index] < 0) {
    neg_set_.insert(index, delta_[index]);
  }

  after_delta_updated(index);

  flip_bit_add_delta(index, -delta_[index]);

  ++flip_count_;
}

inline void SolDelta::greedy() {
  while (neg_count() > 0) {
    vindex_t min_index = neg_min();
    flip(min_index);
  }
  set_if_better("Easy");
}

inline void SolDelta::random_flip(size_t iteration) {
  for (vindex_t i = 0; i < iteration; ++i) {
    vindex_t flip_index = qbpp::misc::RandomEngine::gen(var_count());
    flip(flip_index);
    set_if_better("Easy");
  }
}

inline void SolDelta::move_to(qbpp::Sol destination) {
  misc::MinSet to_be_flipped;
  for (vindex_t i = 0; i < var_count(); ++i) {
    if (get(i) != destination.get(i)) {
      to_be_flipped.insert(i, delta_[i]);
    }
  }
  while (to_be_flipped.var_count() > 0) {
    vindex_t min_index = to_be_flipped.get_first();
    for (vindex_t j = 0; j < quad_model_.degree(min_index); ++j) {
      auto pair = quad_model_.quadratic(min_index, j);
      if (get(pair.first) != destination.get(pair.first)) {
        to_be_flipped.erase(pair.first, delta_[pair.first]);
      }
    }
    to_be_flipped.erase(min_index, delta_[min_index]);
    flip(min_index);
    for (vindex_t j = 0; j < quad_model_.degree(min_index); ++j) {
      auto pair = quad_model_.quadratic(min_index, j);
      if (get(pair.first) != destination.get(pair.first))
        to_be_flipped.insert(pair.first, delta_[pair.first]);
    }
    set_if_better("Easy");
  }
}

inline std::optional<double>
SolDelta::set_if_better(const std::string &solver_name) {
  std::optional<double> tts =
      sol_holder_ptr_->set_if_better(*this, solver_name);
  if (tts.has_value()) {
    easy_solver_.callback(*this, tts.value());
  }
  if (best_sols_ptr() != nullptr) {
    best_sols_ptr()->insert_if_better(*this);
  }
  return tts;
}


inline PosMinSolDelta::PosMinSolDelta(const EasySolver &easy_solver)
    : TabuSolDelta(easy_solver, std::min(static_cast<vindex_t>(5),
                                         easy_solver.var_count() / 5)),
      pos_set_(var_count()) {
  for (vindex_t i = 0; i < var_count(); ++i) {
    if (delta_[i] < 0) {
      neg_set_.insert(i, delta_[i]);
    } else if (delta_[i] > 0) {
      pos_set_.insert(i, delta_[i]);
    }
  }
}

inline void PosMinSolDelta::search(size_t iteration) {
  greedy();
  for (vindex_t i = 0; i < iteration; ++i) {
    std::optional<vindex_t> flip_index = std::nullopt;
    for (vindex_t j = 0; j < tabu_size_ * 2; ++j) {
      vindex_t candidate = vindex_limit;
      if (pos_count() == 0 || neg_count() == 0) {
        candidate = qbpp::misc::RandomGenerator::gen(var_count());
      } else if (qbpp::misc::RandomGenerator::gen(neg_count() + 1) == 0) {
        candidate = pos_min();
      } else {
        candidate = neg_random();
      }
      if (candidate == vindex_limit) {
        throw std::runtime_error(THROW_MESSAGE("Unexpected error."));
      }
      if (!tabu_.has(candidate)) {
        flip_index = candidate;
        break;
      }
    }
    if (!flip_index.has_value()) {
      flip_index = tabu_.non_tabu_random();
    }
    flip(flip_index.value());
    tabu_.insert(flip_index.value());
    set_if_better("Easy");
    if (easy_solver_.get_target_energy().has_value() &&
        sol_holder_ptr_->energy() <= easy_solver_.get_target_energy())
      break;
  }
  greedy();
}

inline void EasySolver::single_search(size_t thread_id) {
  const size_t min_flip_count = 100;
  PosMinSolDelta pos_min_sol_delta(*this);
  size_t max_flip_count = (var_count() / 2);
  if (thread_count_ > 1) {
    double k = std::pow(static_cast<double>(max_flip_count) / min_flip_count,
                        1.0 / static_cast<double>(thread_count_ - 1));
    max_flip_count =
        static_cast<size_t>(min_flip_count * std::pow(k, thread_id));
    if (max_flip_count < min_flip_count)
      max_flip_count = min_flip_count;
  }
  energy_t prev_energy = sol_holder_ptr_->energy();
  size_t random_flip_count = 1;
  while (!time_limit_.has_value() ||
         qbpp::get_time() - start_time_ < time_limit_.value()) {
    pos_min_sol_delta.move_to(sol_holder_ptr_->get_sol());
    pos_min_sol_delta.random_flip(random_flip_count);
    pos_min_sol_delta.search(max_flip_count);
    if (target_energy_.has_value() &&
        sol_holder_ptr_->energy() <= target_energy_.value())
      break;
    if (prev_energy == sol_holder_ptr_->energy()) {
      random_flip_count = (random_flip_count + 1) % (max_flip_count / 2);
      if (random_flip_count == 0)
        random_flip_count = 1;
    } else {
      random_flip_count = 1;
    }
    prev_energy = sol_holder_ptr_->energy();
  }

  {
    std::lock_guard<std::mutex> lock(flip_count_mutex_);
    flip_count_ += pos_min_sol_delta.get_flip_count();
  }
}

inline qbpp::Sol EasySolver::search(bool has_initial_sol) {
  if (!has_initial_sol && is_internal_sol_holder_) {
    sol_holder_ptr_ = std::make_shared<qbpp::SolHolder>(quad_model_);
  }
  start_time_ = qbpp::get_time();
  if (quad_model_.term_count() == 0)
    return sol_holder_ptr_->get_sol();
  prev_energy_ = std::nullopt;

  tbb::parallel_for(size_t(0), static_cast<size_t>(thread_count_),
                    [this](size_t thread_id) { single_search(thread_id); });
  return sol_holder_ptr_->get_sol();
}

} 
} 


#endif 
