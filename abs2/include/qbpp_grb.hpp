/// @file  qbpp_grb.hpp
/// @brief QUBO++ interface to call Gurobi Optimizer
/// @details This file provides interfaces to call Gurobi Optimizer from QUBO++
/// @author Koji Nakano
/// @version 2025-04-21

#ifndef GUROBI_QBPP_HPP
#define GUROBI_QBPP_HPP

#include <cmath>

#include "abs2.hpp"
#include "gurobi_c++.h"
#include "qbpp.hpp"

#define GRB_SAFE_CALL(func)                                               \
  try {                                                                   \
    func;                                                                 \
  } catch (GRBException e) {                                              \
    std::cerr << e.getErrorCode() << ": " << e.getMessage() << std::endl; \
    exit(1);                                                              \
  }

/// @brief Namespace to use Gurobi optimizer from QUBO++ library
namespace qbpp_grb {

//===========================
// Class forward declarations
//===========================
class QuadModel;
class Sol;
class Callback;

//===================
// Class declarations
//===================

/// @brief Class to store a QUBO model using Gurobi Optimizer through QUBO++
/// library
/// @details This class is used to store a QUBO model and solve it using
/// Gurobi Optimizer through QUBO++ library.
class QuadModel : public qbpp::QuadModel {
 protected:
  /// @brief Gurobi environment
  GRBEnv grb_env;
  /// @brief Gurobi model
  std::shared_ptr<GRBModel> grb_model_ptr;
  /// @brief Pointer to Gurobi variables
  GRBVar *grb_x;

 public:
  /// @defgroup qbpp_grb QUBO++ API for Gurobi Optimizer
  /// @brief API to call Gurobi Optimizer from QUBO++ Library
  /// @details API to provide a way to call Gurobi Optimizer from QUBO++
  /// Library.
  /// @{

  /// @brief Constructor to create a model from QuadModel object
  /// @param quad_model QUBO model of QUBO++ library
  /// @param verbose true if Gurobi optimizer outputs are shown on the screen
  QuadModel(const qbpp::QuadModel &quad_model, bool verbose = false);

  /// @brief Copy constructor to create a model from another model
  /// @param grb_model Gurobi model
  QuadModel(const QuadModel &grb_model) = default;

  /// @brief Sets a parameter to the Gurobi environment
  /// @param key parameter name
  /// @param val parameter value
  void set(const std::string &key, const std::string &val) {
    grb_model_ptr->getEnv().set(key, val);
  };

  /// @brief Sets time limit to the Gurobi model
  /// @param time_limit Time limit in seconds
  void set_time_limit(uint32_t time_limit) {
    set("TimeLimit", std::to_string(time_limit));
  }

  /// @brief Set a callback function to the Gurobi model
  /// @param cb Pointer to the class object that includes user-defined
  /// callback function
  /// @details The callback function in the class object *cb is called during
  /// the optimization process of Gurobi Optimizer.
  void set(Callback &cb);

  /// @}

  /// @brief Get the Gurobi variable from the index
  /// @param index index of the variable from 0 to size()-1
  /// @return Gurobi variable
  GRBVar get_grb_var(int index) const { return grb_x[index]; };

  /// @brief Get the Gurobi model
  /// @return Gurobi model
  GRBModel &get_grb_model() { return *grb_model_ptr; }

  /// @brief Get the Gurobi model
  /// @return Gurobi model
  const GRBModel &get_grb_model() const { return *grb_model_ptr; }

  ///@{
  /// @brief Optimize the QUBO model
  /// @return Solution of the QUBO model
  Sol optimize();
  ///@}

  /// @brief Write the model to a file
  /// @param filename File name
  /// @details The model is written in the LP format. The format type is
  /// determined by the file extension such as .lp and .mps.
  void write(std::string filename) {
    GRB_SAFE_CALL(grb_model_ptr->write(filename));
  }
};

/// @brief Class to store a solution of a QUBO model using Gurobi Optimizer
/// @details This class is used to store a solution of a QUBO model obtained
/// by Gurobi Optimizer through QUBO++ library.
class Sol : public qbpp::Sol {
  /// @brief Energy Bound obtained by of the Grobi Optimizer
  qbpp::energy_t bound;

 public:
  /// @defgroup qbpp_grb
  /// @{

  /// @brief Creates a solution object from a model
  /// @param quad_model QUBO model of QUBO++ library
  Sol(const qbpp::QuadModel &quad_model) : qbpp::Sol(quad_model) {}

  /// @}

  /// @brief Gets the energy bound of the solution
  /// @return Energy bound
  qbpp::energy_t get_bound() const { return bound; }

  /// @brief Sets the energy bound of the solution
  /// @return Energy bound
  qbpp::energy_t set_bound(qbpp::energy_t eval) {
    bound = eval;
    return bound;
  }
};

/// @brief Class to manage a callback function called by Gurobi Optimizer
/// @details This class is used to manage a callback function for Gurobi
/// Optimizer through QUBO++ library.
class Callback : public GRBCallback {
 protected:
  /// @brief QUBO model
  const QuadModel quad_model;

  /// @brief Shortcut to the GRBModel in QuadModel.
  const GRBModel &grb_model;

  /// @brief Target energy to stop the Gurobi optimizer.
  std::optional<qbpp::energy_t> target_energy = std::nullopt;

  //  std::chrono::high_resolution_clock::time_point start_time =
  //      std::chrono::high_resolution_clock::now();

  /// @brief Mutex to lock the critical section in get_sol()
  mutable std::mutex mtx;

  /// @brief Abort the optimization process if the target energy is achieved
  /// @param energy Energy of the current solution
  /// @details This function is intended to be called in callback() function.
  void abort_if_target_energy(qbpp::energy_t energy) {
    if (target_energy.has_value() && energy <= target_energy) {
      abort();
    }
  }

 public:
  /// @brief Constructor: a new Callback object
  /// @param quad_model QUBO model of QUBO++ library
  Callback(const QuadModel &quad_model)
      : quad_model(quad_model), grb_model(quad_model.get_grb_model()) {
    qbpp::get_time();
  };

  virtual ~Callback() = default;

  /// @brief Get the solution obtained by Gurobi Optimizer
  /// @return Solution by Gurobi Optimizer
  Sol get_sol();

  /// @brief Default callback function for Gurobi Optimizer
  /// @details This function is called during the optimization process of
  /// Gurobi Optimizer. The function can be customized by overriding it.
  virtual void callback() override {
    if (where == GRB_CB_MIPSOL) {
      qbpp::energy_t energy = get_sol().energy();
      std::cout << "TTS = " << std::fixed << std::setprecision(3)
                << std::setfill('0') << qbpp::get_time()
                << "s Energy = " << energy << std::endl;
      abort_if_target_energy(energy);
    }
  }

  /// @brief Set the target energy for Gurobi Optimizer
  void set_target_energy(qbpp::energy_t target_energy) {
    this->target_energy = target_energy;
  }

  /// @brief Calls GetDoubleInfo() of GRBCallback
  /// @param what what to get
  /// @return double value
  double getDoubleInfoPublic(int what) { return getDoubleInfo(what); }

  /// @brief Calls getSolution() of GRBCallback
  /// @param v Gurobi variable
  /// @return double value
  double getSolutionPublic(GRBVar v) { return getSolution(v); }
};

//=============================
// Class QuadModel member functions
//=============================

inline QuadModel::QuadModel(const qbpp::QuadModel &quad_model, bool verbose)
    : qbpp::QuadModel(quad_model), grb_env(true) {
  // Suppress Gurobi outputs to the screen.
  grb_env.set(GRB_IntParam_OutputFlag, 0);
  if (verbose) {
    grb_env.set("OutputFlag", "1");
  }
  GRB_SAFE_CALL(grb_env.start());

  grb_model_ptr = std::make_unique<GRBModel>(grb_env);

  //  Objective is minimization.
  GRB_SAFE_CALL(grb_model_ptr->set(GRB_IntAttr_ModelSense, GRB_MINIMIZE));

  // Add binary variables to the Gurobi model.
  grb_x = grb_model_ptr->addVars(var_count(), GRB_BINARY);
  // Compute objective function of QUBO problem.
  GRBQuadExpr obj;
  obj += quad_model.constant();
  for (qbpp::vindex_t i = 0; i < var_count(); ++i) {
    if (quad_model.linear(i) != 0) obj += quad_model.linear(i) * grb_x[i];
  }
  for (qbpp::vindex_t i = 0; i < var_count(); ++i) {
    for (qbpp::vindex_t j = 0; j < quad_model.degree(i); ++j) {
      auto [k, coeff] = quad_model.quadratic(i, j);
      if (i < k) obj += coeff * grb_x[i] * grb_x[k];
    }
  }

  // Set the objective function to the Gurobi model.
  GRB_SAFE_CALL(grb_model_ptr->setObjective(obj));
}

inline void QuadModel::set(Callback &cb) { grb_model_ptr->setCallback(&cb); }

inline Sol QuadModel::optimize() {
  Sol sol(*this);
  GRB_SAFE_CALL(grb_model_ptr->optimize());
  sol.set_energy(std::round(grb_model_ptr->get(GRB_DoubleAttr_ObjVal)));
  sol.set_bound(std::round(grb_model_ptr->get(GRB_DoubleAttr_ObjBound)));
  for (qbpp::vindex_t i = 0; i < var_count(); ++i)
    sol.set(i, int(grb_x[i].get(GRB_DoubleAttr_X)));
  return sol;
}

//================================
// Class Callback member functions
//================================

inline Sol Callback::get_sol() {
  Sol sol(quad_model);
  std::lock_guard<std::mutex> lock(mtx);
  // Intentionally commented out because the solution obtained by getSolution
  // may provide corrupted values when Gurobi finds an optimal solution.
  // sol.set_energy(std::round(getDoubleInfoPublic(GRB_CB_MIPSOL_OBJ)));
  sol.set_bound(std::round(getDoubleInfoPublic(GRB_CB_MIPSOL_OBJBND)));
  // Get the solution from Gurobi Optimizer.
  for (qbpp::vindex_t i = 0; i < quad_model.var_count(); ++i) {
    sol.set(i, int(getSolutionPublic(quad_model.get_grb_var(i))));
  }
  // eval is used to compute the energy of the solution instead of MIPSOL_OBJ.
  sol.set_energy(eval(quad_model, sol));
  return sol;
}

}  // namespace qbpp_grb
#endif  // __GUROBI_QBPP_HPP__
