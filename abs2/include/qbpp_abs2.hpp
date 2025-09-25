/// @file qbpp_abs2.hpp
/// @brief	QUBO++ interface to call ABS2 GPU QUBO solver
/// @details This file provides interfaces to call ABS2 QUBO solver from QUBO++
/// @author Koji Nakano
/// @version 2025-04-21

#ifndef QBPP_ABS2_HPP
#define QBPP_ABS2_HPP

#include <algorithm>

#include "abs2.hpp"
#include "qbpp.hpp"

/// @brief Namespace to use ABS2 QUBO solver from QUBO++ library
namespace qbpp_abs2 {

//===========================
// Class forward declarationsmake
//===========================
class Solver;
class QuadModel;
class Param;
class Sol;
class Callback;

//===================
// Class declarations
//===================
/// @defgroup qbpp_abs2 QUBO++ API for ABS2 QUBO Solver
/// @brief API for calling the ABS2 QUBO solver from the QUBO++ Library
/// @details This API provides a way to call the ABS2 QUBO solver from the
/// QUBO++ Library. It includes the following classes:
/// - Solver: A class for calling the ABS2 QUBO solver, derived from
/// abs2::Solver.
/// - QuadModel: A class for storing both the qbpp::QuadModel and abs2::Model,
/// derived from qbpp::QuadModel and containing an abs2::Model object.
/// - Param: A class for setting parameters for the ABS2 QUBO solver, derived
/// from abs2::Param.
/// - Sol: A class for storing the ABS2 QUBO solution, derived from qbpp::Sol
/// and containing an abs2::Sol object.
/// - Callback: A class for defining the ABS2 callback function, derived from
/// abs2::Callback and containing a qbpp::QuadModel object.
///
/// Please refer to [Sample Programs](SAMPLE.md) for examples of how
/// to use these classes.

/// @{

/// @brief A class for calling the ABS2 QUBO solver.
/// @details This class is derived from derived from abs2::Solver.
/// @note This class is inherited from abs2::Solver.
class Solver : public abs2::Solver {
 public:
  /// @brief Executes ABS2 for "quad_model" with "param" and returns "sol".
  /// @param quad_model QuadModel object
  /// @param param ABS2 parameters
  /// @return Sol QUBO solution
  Sol operator()(const QuadModel &quad_model, Param &param) const;

  /// @brief  Executes ABS2 for model with "param" and "start", and returns
  /// "sol".
  /// @param quad_model QuadModel object
  /// @param param ABS2 parameters
  /// @param start Initial solution
  /// @return Sol QUBO solution
  Sol operator()(const QuadModel &quad_model, Param &param,
                 const Sol &start) const;

  /// @}
};

/// @brief A class for storing both the qbpp::QuadModel and abs2::Model.
/// @details This class is derived from qbpp::QuadModel and containing an
/// abs2::Model object.
class QuadModel : public qbpp::QuadModel {
  /// @brief ABS2 model associated with the QUBO model
  std::shared_ptr<abs2::Model> abs2model_ptr;

  /// @brief Default constructor is deleted to avoid creating an empty model.
  QuadModel() = delete;

 public:
  /// @brief Constructor: Create an ABS2 model from a QUBO model.
  /// @param quad_model qbpp::QuadModel object
  QuadModel(const qbpp::QuadModel &quad_model);

  /// @brief Copy constructor
  /// @param quad_model QuadModel object
  QuadModel(const QuadModel &quad_model)
      : qbpp::QuadModel(quad_model), abs2model_ptr(quad_model.abs2model_ptr) {};

  /// @brief Get the ABS2 model
  /// @return ABS2 model
  const abs2::Model &get_abs2_model() const { return *abs2model_ptr; }
};

/// @brief A class for setting parameters for the ABS2 QUBO solver.
/// @details This class is derived from abs2::Param.
/// @note This class stores the target_energy qbpp::Expr object, which includes
/// a constant term. Since the ABS2 QUBO solver does not handle constant terms,
/// the target_energy passed to the ABS2 solver must be adjusted and set
/// accordingly when the ABS2 solver is called.
class Param : public abs2::Param {
  /// @brief The target energy given by C++ library.
  /// @note ABS2 QUBO solver has no constant term, the target_value must be set
  /// when ABS2 solver is called.
  std::optional<qbpp::energy_t> target_energy;

 public:
  ///@defgroup qbpp_abs2
  ///@{

  /// @brief Construct a new Param object from a QuadModel object.
  /// @note The default constructor can be omitted but defined for Doxygen.
  Param() = default;

  /// @brief Set the target energy for ABS2 QUBO solver.
  /// @param target_energy Target energy
  void set_target_energy(qbpp::energy_t target_energy) {
    this->target_energy = target_energy;
  }

  /// @brief Get the target energy for ABS2 QUBO solver.
  /// @return Target energy
  std::optional<qbpp::energy_t> get_target_energy() const {
    return target_energy;
  }

  /// @brief Set the time limit for ABS2 QUBO solver.
  /// @param time_limit Time limit in seconds.
  void set_time_limit(uint32_t time_limit) {
    abs2::Param::set("time_limit", std::to_string(time_limit));
  }

  /// @brief Set the arithmetic bits
  /// @param bits Number of bits for arithmetic: 32 or 64
  void set_arithmetic_bits(uint32_t bits) {
    abs2::Param::set("arithmetic_bits", std::to_string(bits));
  }

  ///@}
};

/// @brief A class for storing the ABS2 QUBO solution
/// @details This class is derived from qbpp::Sol and containing an abs2::Sol
/// object.
class Sol : public qbpp::Sol {
  /// @brief Sol object created by ABS2 QUBO solver.
  /// @note The shared_ptr is used to avoid copying the object.
  const std::shared_ptr<abs2::Sol> abs2sol_ptr;

 public:
  /// @defgroup qbpp_abs2
  /// @{

  /// @brief Construct a new Sol object from an ABS2 solution.
  /// @param quad_model ABS2 QUBO model
  /// @param sol ABS2 solution
  Sol(const QuadModel &quad_model, const abs2::Sol &sol);

  /// @brief Construct a new Sol object from a QUBO++ QUBO solution.
  /// @param sol QUBO++ QUBO solution
  Sol(const qbpp::Sol &sol);

  /// @}

  /// @brief Copy constructor
  /// @param sol Sol object
  Sol(const Sol &sol) = default;

  /// @brief print the solution
  /// @param attrs attributes to print
  void print(const std::string &attrs) const { abs2sol_ptr->print(attrs); }

  /// @brief Set the value of the variable with "index" to "value".
  /// @param index Variable index
  /// @param value Value
  void set(int index, bool value) {
    qbpp::Sol::set(index, value);
    abs2sol_ptr->set(index, value);
  }

  /// @brief Set the value of the variable "var" to "value".
  /// @param var Variable
  /// @param value Value
  void set(qbpp::Var var, bool value) {
    qbpp::Sol::set(var, value);
    abs2sol_ptr->set(index(var), value);
  }

  double get_tts() const { return std::stod(abs2sol_ptr->get("tts")); }

  /// @brief Returns the reference to the ABS2 solution.
  /// @return ABS2 solution
  const std::shared_ptr<abs2::Sol> get_abs2sol_ptr() const {
    return abs2sol_ptr;
  }
};

/// @brief A class for defining the ABS2 callback function
/// @details This class is derived from abs2::Callback and containing a
/// qbpp::QuadModel object.
class Callback : public abs2::Callback {
 protected:
  /// @brief QuadModel object created by QUBO++ library.
  const QuadModel quad_model;

 public:
  /// @defgroup qbpp_abs2
  /// @{

  /// @brief Construct a new Callback object
  /// @param quad_model ABS2 QuadModel object
  Callback(const QuadModel &quad_model)
      : abs2::Callback(), quad_model(quad_model) {};

  /// @brief Get the solution from the ABS2 solver.
  /// @return Sol QUBO solution
  Sol get_sol() const { return Sol(quad_model, abs2::Callback::get()); };

  /// @brief Provide a hint solution to the ABS2 solver.
  /// @param hint Hint solution such as the best solution found so far.
  /// @details Hint is given to the ABS2 solver to improve the solution.
  void set_hint(const Sol &hint) {
    abs2::Callback::set(*hint.get_abs2sol_ptr());
  }
  /// @brief The default callback function for ABS2 QUBO solver.
  /// @details This function is called by ABS2 QUBO solver when an event occurs.
  /// @param event Event type specfied by ABS2 QUBO solver
  /// @note This is a default callback function that displays the TTS and the
  /// energy when a new best solution is obtained. This function can be
  /// customized by overriding it.
  virtual void callback(const std::string &event) {
    if (event == "init") {
      qbpp_abs2::Callback::set("new");
    } else if (event == "new") {
      auto sol = get_sol();
      std::cout << "TTS = " << std::fixed << std::setprecision(3)
                << sol.get_tts() << "s Energy=" << sol.energy() << std::endl;
    }
  }

  ///@}

};  // namespace qbpp_abs2

//============================
// Class QuadModel member function
//============================
inline QuadModel::QuadModel(const qbpp::QuadModel &quad_model)
    : qbpp::QuadModel(quad_model) {
  // Create an empty ABS2 model
  abs2model_ptr = std::make_unique<abs2::Model>(
      var_count(), quad_model.min_coeff(), quad_model.max_coeff());

  // Set the linear and quadratic terms
  for (qbpp::vindex_t i = 0; i < var_count(); i++) {
    if (quad_model.linear(i) != 0)
      abs2model_ptr->set(i, i, quad_model.linear(i));
  }
  for (qbpp::vindex_t i = 0; i < var_count(); i++) {
    for (qbpp::vindex_t j = 0; j < quad_model.degree(i); j++) {
      auto [k, coeff] = quad_model.quadratic(i, j);
      abs2model_ptr->set(i, k, coeff);
    }
  }
}

//===============================
// Class Solver member functions
//===============================

// Execute ABS2 for "quad_model" with "param" and returns "sol".
inline Sol Solver::operator()(const QuadModel &quad_model, Param &param) const {
  // target_energy is adjusted by the constant term of the QUBO model because
  // ABS2 Solver does not handle constant terms.
  if (param.get_target_energy().has_value()) {
    param.set("target_energy",
              std::to_string(param.get_target_energy().value() -
                             quad_model.constant()));
  }
  return Sol(quad_model,
             abs2::Solver::operator()(quad_model.get_abs2_model(), param));
}

// Execute ABS2 for model with "quad_param" and "start", and returns "sol".
inline Sol Solver::operator()(const QuadModel &quad_model, Param &param,
                              const Sol &start) const {
  // target_energy is adjusted by the constant term of the QUBO model because
  // ABS2 Solver does not handle constant terms.
  if (param.get_target_energy().has_value()) {
    param.set("target_energy",
              std::to_string(param.get_target_energy().value() -
                             quad_model.constant()));
  }
  return Sol(quad_model,
             abs2::Solver::operator()(quad_model.get_abs2_model(), param,
                                      *start.get_abs2sol_ptr()));
}

//=============================
// Class Sol member functions
//=============================
inline Sol::Sol(const QuadModel &quad_model, const abs2::Sol &sol)
    : qbpp::Sol(quad_model), abs2sol_ptr(std::make_shared<abs2::Sol>(sol)) {
  for (qbpp::vindex_t i = 0; i < quad_model.var_count(); i++) {
    qbpp::Sol::set(i, abs2sol_ptr->get(i));
  }
  energy_ = std::stod(abs2sol_ptr->get("energy")) + quad_model.constant();
}

inline Sol::Sol(const qbpp::Sol &sol)
    : qbpp::Sol(sol), abs2sol_ptr(std::make_shared<abs2::Sol>(var_count())) {
  for (qbpp::vindex_t i = 0; i < var_count(); i++) {
    abs2sol_ptr->set(i, sol.get(i));
  }
}

}  // namespace qbpp_abs2
#endif  // QBPP_ABS2_HPP