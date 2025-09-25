/// @file qbpp_misc.hpp
/// @author Koji Nakano
/// @brief A miscellaneous library used for sample programs of the QUBO++
/// library.
/// @details This library includes miscellaneous classes and functions used for
/// sample programs of the QUBO++ library. The classes include random number
/// generators and graph drawing functions.
/// @copyright 2025, Koji Nakano
/// @version 2025-04-21

#ifndef QBPP_MISC_HPP
#define QBPP_MISC_HPP

#include <algorithm>
#include <atomic>
#include <boost/circular_buffer.hpp>
#include <boost/random/taus88.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "qbpp.hpp"

namespace qbpp {

namespace misc {

#include <boost/random/taus88.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

class RandomEngine {
  boost::random::taus88 rng;

  explicit RandomEngine(uint32_t base_seed = 0) {
    uint32_t rd_seed = static_cast<uint32_t>(std::random_device{}());
    rng.seed(rd_seed ^ base_seed);
  }

  uint32_t gen32_impl() { return rng(); }

  uint64_t gen64_impl() { return (static_cast<uint64_t>(rng()) << 32) | rng(); }

  template <typename T>
  T gen_impl(T n) {
    if constexpr (sizeof(T) <= 4) {
      boost::random::uniform_int_distribution<T> dist(0, n - 1);
      return dist(rng);
    } else {
      struct Engine64 {
        boost::random::taus88 &base;
        using result_type = uint64_t;
        static constexpr result_type min() { return 0; }
        static constexpr result_type max() { return UINT64_MAX; }
        result_type operator()() {
          return (static_cast<uint64_t>(base()) << 32) | base();
        }
      } engine64{rng};
      boost::random::uniform_int_distribution<T> dist(0, n - 1);
      return dist(engine64);
    }
  }

  double gen_double_impl() {
    boost::random::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
  }

  static RandomEngine &instance() {
    thread_local static RandomEngine engine(assign_thread_id());
    return engine;
  }

  static uint32_t assign_thread_id() {
    static std::atomic<uint32_t> counter{0};
    return counter++;
  }

 public:
  static uint32_t gen32() { return instance().gen32_impl(); }

  static uint64_t gen64() { return instance().gen64_impl(); }

  template <typename T>
  static T gen(T n) {
    return instance().gen_impl(n);
  }

  static double gen_double() { return instance().gen_double_impl(); }
};

class RandomGenerator {
  std::mt19937_64 mt;

  RandomGenerator() : mt(std::random_device{}()) {}

  RandomGenerator(const RandomGenerator &) = delete;

  RandomGenerator &operator=(const RandomGenerator &) = delete;

  static RandomGenerator &get_instance() {
    static RandomGenerator instance;
    return instance;
  }

 public:
  static void set_seed(uint32_t seed = 1) { get_instance().mt.seed(seed); }

  static void rd_seed() { get_instance().mt.seed(std::random_device{}()); }

  static uint64_t gen() { return static_cast<uint64_t>(get_instance().mt()); }

  template <typename T>
  static typename std::enable_if<std::is_integral<T>::value, T>::type gen(T n) {
    std::uniform_int_distribution<qbpp::vindex_t> dist(0, n - 1);
    return dist(get_instance().mt);
  }

  static qbpp::cpp_int gen(const qbpp::cpp_int &n) {
    qbpp::cpp_int val;
    do {
      val = (val << 64) + gen();
    } while (val < (n << 32));
    return val % n;
  }

  static double gen_double() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(get_instance().mt);
  }

  static std::mt19937_64 &get_mt() { return get_instance().mt; }
};

class RandomPermutation {
  const uint32_t size_;

  std::vector<uint32_t> perm;

  uint32_t index = 0;

 public:
  explicit RandomPermutation(uint32_t size) : size_(size), perm(size) {
    std::iota(perm.begin(), perm.end(), 0);
  }

  uint32_t get() {
    if (index == 0) {
      std::shuffle(perm.begin(), perm.end(), RandomGenerator::get_mt());
    }
    uint32_t current = index;
    index = (index + 1) % size_;
    return perm[current];
  }
};

struct PcloseDeleter {
  void operator()(FILE *file) const {
    if (file) pclose(file);
  }
};


class RandomSet {
  std::vector<vindex_t> position_;

  std::vector<vindex_t> variables_;

 public:
  explicit RandomSet(vindex_t size) : position_(size, vindex_limit) {
    variables_.reserve(size);
  }

  vindex_t var_count() const {
    return static_cast<vindex_t>(variables_.size());
  }

  void insert(vindex_t index) {
    if (position_[index] != vindex_limit) {
      throw std::runtime_error(THROW_MESSAGE("RandomSet: Insert variable (",
                                             index, ") already in set"));
    }
    variables_.push_back(index);
    position_[index] = static_cast<vindex_t>(variables_.size() - 1);
  }

  void swap(vindex_t pos1, vindex_t pos2) {
    if (pos1 == pos2) return;
    std::swap(variables_[pos1], variables_[pos2]);
    position_[variables_[pos1]] = pos1;
    position_[variables_[pos2]] = pos2;
  }

  void erase(vindex_t index) {
    if (position_[index] == vindex_limit) {
      throw std::runtime_error(
          THROW_MESSAGE("RandomSet: Erase variable (", index, ") not in set"));
    }
    swap(position_[index], var_count() - 1);
    variables_.pop_back();
    position_[index] = vindex_limit;
  }

  bool has(vindex_t index) const { return position_[index] != vindex_limit; }

  vindex_t select_at_random() const {
    return variables_[qbpp::misc::RandomGenerator::gen(var_count())];
  }

  void print(const std::string &prefix = "") const {
    std::cout << prefix << " Size = " << var_count() << " : ";
    for (vindex_t i = 0; i < var_count(); ++i) {
      std::cout << " " << variables_[i];
    }
    std::cout << std::endl;
  }
};

template <typename T = energy_t>
class MinSet {

  using ValIndexMap = std::set<std::pair<T, vindex_t>>;

  ValIndexMap set_;

 public:
  MinSet() = default;

  vindex_t var_count() const { return static_cast<vindex_t>(set_.size()); }

  void insert(vindex_t index, T delta) {
    auto tuple_val = std::make_pair(delta, index);
    auto result = set_.insert(tuple_val);
    if (!result.second) {
      throw std::runtime_error(THROW_MESSAGE("MinSet: Insert variable (", index,
                                             ") already in set"));
    }
  }

  void erase(vindex_t index, T delta) {
    auto pair_val = std::make_pair(delta, index);
    auto result = set_.erase(pair_val);
    if (result == 0) {
      throw std::runtime_error(
          THROW_MESSAGE("MinSet: Erase variable (", index, ") not in set"));
    }
  }

  vindex_t get_first() const { return (*set_.begin()).second; }

  std::pair<T, vindex_t> get_min() const { return *set_.begin(); }

  bool empty() const { return set_.empty(); }

  void print(const std::string &prefix) const {
    std::cout << prefix;
    for (auto &[val, i] : set_) {
      if constexpr (std::is_same_v<T, energy_t>) {
        std::cout << "(" << i << "," << val << ")";
      } else {
        std::cout << "(" << i << "," << val.get_constraint_delta() << ","
                  << val.get_objective_delta() << ")";
      }
    }
    std::cout << std::endl;
  }
};

template <typename T = energy_t>
class MinHeap {
  std::vector<std::pair<T, vindex_t>> heap_;
  std::vector<vindex_t> index_;

  void emplace_back(vindex_t index, energy_t delta) {
    heap_.emplace_back(std::make_pair(delta, index));
    index_[index] = heap_size() - 1;
  }

  void swap_heap(vindex_t i, vindex_t j) {
    std::swap(heap_[i], heap_[j]);
    index_[heap_[i].second] = i;
    index_[heap_[j].second] = j;
  }

  void bubble_up(vindex_t i) {
    while (i > 0) {
      vindex_t parent = (i - 1) / 2;
      if (heap_[i] < heap_[parent]) {
        swap_heap(i, parent);
        i = parent;
      } else {
        break;
      }
    }
  }

  void bubble_down(vindex_t i) {
    while (true) {
      vindex_t left = 2 * i + 1;
      vindex_t right = 2 * i + 2;
      vindex_t smallest = i;

      if (left < heap_size() && heap_[left] < heap_[smallest]) {
        smallest = left;
      }
      if (right < heap_size() && heap_[right] < heap_[smallest]) {
        smallest = right;
      }
      if (smallest != i) {
        swap_heap(i, smallest);
        i = smallest;
      } else {
        break;
      }
    }
  }

 public:
  MinHeap(size_t size) : index_(size, vindex_limit) {}

  vindex_t heap_size() const { return static_cast<vindex_t>(heap_.size()); }
  vindex_t get_first() const { return heap_[0].second; }

  bool has(vindex_t index) const { return index_[index] != vindex_limit; }

  bool empty() const { return heap_.empty(); }

  vindex_t select_at_random() const {
    return heap_[qbpp::misc::RandomGenerator::gen(heap_size())].second;
  }

  void insert(vindex_t index, energy_t delta) {
    if (index_[index] != vindex_limit) {
      throw std::runtime_error(THROW_MESSAGE("DualMinHeap: Insert variable (",
                                             index,
                                             ") already in "
                                             "heap"));
    }
    emplace_back(index, delta);
    bubble_up(heap_size() - 1);
  }

  void erase(vindex_t index) {
    vindex_t i = index_[index];
    if (i == vindex_limit) {
      throw std::runtime_error(THROW_MESSAGE("DualMinHeap: Erase variable (",
                                             index,
                                             ") not in "
                                             "heap"));
    }
    swap_heap(i, heap_size() - 1);
    heap_.pop_back();
    index_[index] = vindex_limit;
    if (i < heap_size()) {
      bubble_up(i);
      bubble_down(i);
    }
  }

  void print(const std::string &prefix) const {
    std::cout << prefix;
    for (auto &[delta, i] : heap_) {
      std::cout << "(" << i << "," << delta << ")";
    }
    std::cout << std::endl;
  }
};

template <typename T = energy_t>
class RandomMinSet {
  MinSet<T> min_set_;
  RandomSet random_set_;

 public:
  RandomMinSet(vindex_t size) : random_set_(size) {};

  vindex_t var_count() const {
    return static_cast<vindex_t>(min_set_.var_count());
  }

  void insert(vindex_t index, T delta) {
    min_set_.insert(index, delta);
    random_set_.insert(index);
  }

  void erase(vindex_t index, T delta) {
    min_set_.erase(index, delta);
    random_set_.erase(index);
  }

  bool has(vindex_t index) const { return random_set_.has(index); }

  vindex_t select_at_random() const { return random_set_.select_at_random(); }

  vindex_t get_first() const { return min_set_.get_first(); }

  std::pair<T, vindex_t> get_min() const { return min_set_.get_min(); }

  bool empty() const { return min_set_.empty(); }

  void print(const std::string &prefix) const {
    min_set_.print(prefix + " MIN:");
    random_set_.print(prefix + "RANDOM:");
  }
};

class Tabu {

  const vindex_t var_count_;

  std::unordered_set<vindex_t> tabu_set_;

  boost::circular_buffer<vindex_t> tabu_list;

 public:
  Tabu(vindex_t var_count, vindex_t tabu_capacity)
      : var_count_(var_count), tabu_list(tabu_capacity) {}

  void set_capacity(vindex_t capacity) {
    tabu_set_.clear();
    tabu_list = boost::circular_buffer<vindex_t>(capacity);
  }

  vindex_t size() const { return static_cast<vindex_t>(tabu_list.size()); }

  vindex_t get_capacity() const {
    return static_cast<vindex_t>(tabu_list.capacity());
  }

  bool has(vindex_t index) const {
    return tabu_set_.find(index) != tabu_set_.end();
  }

  std::optional<vindex_t> insert(vindex_t index, bool must_be_new) {
    std::optional<vindex_t> removed = std::nullopt;

    if (tabu_list.capacity() == 0) {
      return index;
    }
    if (has(index)) {
      if (must_be_new) {
        throw std::runtime_error(THROW_MESSAGE("Tabu: Insert variable (", index,
                                               ") already in tabu list"));

      } else {
        auto it = std::find(tabu_list.begin(), tabu_list.end(), index);
        tabu_list.erase(it);
      }
    }
    if (tabu_list.full()) {
      removed = tabu_list.front();
      tabu_set_.erase(*removed);
      tabu_list.pop_front();
    }
    tabu_set_.insert(index);
    tabu_list.push_back(index);
    return removed;
  }

  std::optional<vindex_t> insert(vindex_t index) {
    return insert(index, false);
  }

  std::optional<vindex_t> insert_new(vindex_t index) {
    return insert(index, true);
  }

  std::optional<vindex_t> erase_front() {
    std::optional<vindex_t> removed = std::nullopt;
    if (size() == 0) {
      return removed;
    }
    removed = tabu_list.front();
    tabu_set_.erase(*removed);
    tabu_list.pop_front();
    return removed;
  }

  vindex_t non_tabu_random() {
    vindex_t index;
    do {
      index = qbpp::misc::RandomEngine::gen(var_count_);
    } while (has(index));
    return index;
  }

  void print() {
    std::cout << "Tabu list (capacity = " << tabu_list.capacity() << "):";
    for (const auto &i : tabu_list) {
      std::cout << " " << i;
    }
    std::cout << std::endl;
  }
};

}  
}  

#endif  