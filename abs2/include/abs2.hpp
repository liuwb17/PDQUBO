/// @file abs2.hpp
/// @author Koji Nakano
/// @brief API for the ABS2 GPU QUBO Solver.
/// @details This header file provides API definitions for interfacing with the
/// ABS2 GPU QUBO solver. ABS2 is a QUBO (Quadratic Unconstrained Binary
/// Optimization) solver that leverages GPU capabilities to efficiently solve
/// QUBO problems.
///
/// Please refer to the following article for more information on the ABS2 GPU
/// QUBO solver:
/// Koji Nakano et al.,Diverse Adaptive Bulk Search: a Framework for Solving
/// QUBO Problems on Multiple GPUs. IPDPS Workshops 2023: 314-325.
/// https://doi.org/10.1109/IPDPSW59300.2023.00060
///
/// @note The source code for ABS2 is not available. Only the executable shared
/// library for the ABS2 GPU QUBO solver is provided.
/// @version 2024-12-29

#ifndef ABS2_HPP
#define ABS2_HPP
#include <iostream>
#include <memory>
#include <string>

///@brief Namespace to call ABS2 GPU QUBO solver
namespace abs2 {
//===========================
// Class forward declarations
//===========================
class Solver;
class Model;
class Param;
class Sol;
class Callback;

//====================
// Class declarations
//====================

/// @brief Class to configure the ABS2 QUBO solver.
/// @details This class is used to check available GPUs and verify the
/// installation of the ABS2 QUBO solver shared libraries.
class Solver {
 public:
  /// @brief Constructor for the ABS2 QUBO solver.
  /// @param verbose If true, displays license information.
  Solver(bool verbose = true);

  /// @brief Destructor for the ABS2 QUBO solver.
  ~Solver();

  /// @brief Retrieves the solver attribute value associated with the given key.
  /// @param key The attribute key.
  /// @return The attribute value as a string.
  std::string get(const std::string &key);

  /// @brief Outputs the solver attribute value in the specified format.
  /// @param option The format option.
  /// @param output_stream The output stream to write to. Defaults to std::cout.
  void print(const std::string &option = "",
             std::ostream &output_stream = std::cout) const;

  /// @brief Outputs the solver attribute value in the specified format.
  /// @param output_stream The output stream to write to.
  /// @param option The format option. Defaults to an empty string.
  void print(std::ostream &output_stream, const std::string &option = "") const;

  /// @brief Executes the ABS2 solver for the given model and parameters,
  /// returning the solution.
  /// @param model The QUBO model.
  /// @param param The ABS2 solver parameters.
  /// @return The solution as a Sol object.
  Sol operator()(const Model &model, const Param &param) const;

  /// @brief Executes the ABS2 solver for the given model, parameters, and
  /// initial solution, returning the solution.
  /// @param model The QUBO model.
  /// @param param The ABS2 solver parameters.
  /// @param start The initial solution.
  /// @return The solution as a Sol object.
  Sol operator()(const Model &model, const Param &param,
                 const Sol &start) const;

 private:
  class Impl;
  //  std::unique_ptr<Impl> pimpl;
  Impl *pimpl;
};

/// @brief Class to store and manipulate a QUBO model.
class Model {
 public:
  /// @brief Constructor for an empty QUBO model.
  /// @param size Number of variables.
  /// @param bits Number of bits for each coefficient.
  /// @note The "size" must be between 32 and 65536, and [min_coeff, max_coeff]
  /// defines the range of W_{i,j}.
  Model(int size, int bits);

  /// @brief Constructor for an empty QUBO model with specified variable size
  /// and coefficient range.
  /// @param size Number of variables.
  /// @param min_coeff Minimum value of W_{i,j}.
  /// @param max_coeff Maximum value of W_{i,j}.
  Model(int size, int64_t min_coeff, int64_t max_coeff);

  /// @brief Constructor for loading a QUBO model from a file.
  /// @param filename The name of the file to load the model from.
  Model(const std::string &filename);

  /// @brief Destructor for the QUBO model.
  ~Model();

  /// @brief Copy constructor for the QUBO model.
  /// @param model The QUBO model to be copied.
  Model(const Model &model);

  /// @brief Assignment operator for copying the QUBO model.
  /// @param model The QUBO model to be copied.
  /// @return Reference to the copied QUBO model.
  Model &operator=(const Model &model);

  /// @brief Sets the value of W_{i,j} if i<=j, and W_{j,i} if j<i.
  /// @param i Row index of W.
  /// @param j Column index of W.
  /// @param val Value to be set.
  void set(int i, int j, int64_t val);

  /// @brief Gets the value of W_{i,j} if i<=j, and W_{j,i} if j<i.
  /// @param i Row index of W.
  /// @param j Column index of W.
  /// @return The value of W_{i,j} or W_{j,i}.
  int64_t get(int i, int j) const;

  /// @brief Sets a value for a given key.
  /// @param key The key.
  /// @param val The value to be set.
  void set(const std::string &key, const std::string &val);

  /// @brief Gets the value associated with a given key.
  /// @param key The key.
  /// @return The value associated with the key.
  std::string get(const std::string &key) const;

  /// @brief Outputs the model to the specified output stream.
  /// @param option The format option.
  /// @param output_stream The output stream to write to. Defaults to std::cout.
  void print(const std::string &option = "",
             std::ostream &output_stream = std::cout) const;

  /// @brief Outputs the model to the specified output stream.
  /// @param output_stream The output stream to write to.
  /// @param option The format option. Defaults to an empty string.
  void print(std::ostream &output_stream, const std::string &option = "") const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

///@brief Class to store parameters for ABS2 QUBO solver.
class Param {
 public:
  /// @brief Constructor for creating param object with empty parameters.
  Param();
  ///@brief Destructor
  ~Param();
  /// @brief Copy Constructor with copying "param".
  /// @param param Param object to be copied
  Param(const Param &param);
  /// @brief Assignment overload with copying "param".
  /// @param param Param object to be copied
  /// @return Param object
  Param &operator=(const Param &param);
  /// @brief Set "val" to "key".
  /// @param key Parameter key
  /// @param val Parameter value
  void set(const std::string &key, const std::string &val);
  /// @brief Get the value of "key".
  /// @param key Parameter key
  /// @return Parameter value
  std::string get(const std::string &key) const;
  /// @brief Set Callback function to ABS2.
  /// @param callback Callback class object that includes user-defined callback
  /// function
  /// @details The callback function in the class object "callback" is called
  /// during the optimization process of ABS2.
  void set(Callback &callback);
  /// @brief Output parameters in stream "output_stream".
  /// @param option Format option
  /// @param output_stream Output stream
  void print(const std::string &option = "",
             std::ostream &output_stream = std::cout) const;
  /// @brief Output parameters in stream "output_stream".
  /// @param output_stream Output stream
  /// @param option Format option
  void print(std::ostream &output_stream, const std::string &option = "") const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};
/// @brief Class to store a solution computed by ABS2 QUBO solver.
/// @note The size of the solution can be changed after the construction
class Sol {
 public:
  /// @brief Constructor for a solution with "size" bits.
  /// @param size Number of bits
  Sol(int size);
  /// @brief Construct a new Sol object
  Sol();
  ///@brief Destructor
  ~Sol();
  /// @brief Copy constructor. Create a new Sol object from "sol".
  /// @param sol Sol object to be copied
  Sol(const Sol &sol);
  /// @brief Move constructor. Create a new Sol object from "sol".
  /// @param sol Sol object to be moved
  Sol(Sol &&sol);
  /// @brief Copy assignment of "sol"
  /// @param sol Sol object to be copied
  Sol &operator=(const Sol &sol);
  /// @brief Move assignment of "sol"
  /// @param sol Sol object to be moved
  Sol &operator=(Sol &&sol);
  /// @brief Set "val" to x_i.
  /// @param i Index of x
  /// @param val Value to be set
  void set(int i, bool val);
  /// @brief Get the value of x_i.
  /// @param i Index of x
  /// @return x_i
  bool get(int i) const;
  /// @brief Set "val" to key.
  /// @param key Solution key
  /// @param val Solution value
  void set(const std::string &key, const std::string &val);
  /// @brief Get the value of key.
  /// @param key Solution key
  /// @return Solution value
  const std::string get(const std::string &key) const;
  /// @brief Output solution in stream "output_stream".
  /// @param option Format option
  /// @param output_stream Output stream
  void print(const std::string &option = "",
             std::ostream &output_stream = std::cout) const;
  ///  @brief Output solution in stream "output_stream".
  /// @param option Format option
  /// @param output_stream Output stream
  void print(std::ostream &output_stream, const std::string &option = "") const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

///@brief Class to manage callback function for ABS2 QUBO solver.
class Callback {
 public:
  /// @brief Constructor for creating a callback object.
  Callback();
  /// @brief Destructor for deleting a callback object.
  virtual ~Callback();
  /// @brief Callback function must be defined by user.
  /// @param event Event string
  /// @details The callback function must be defined by the user.
  virtual void callback(const std::string &event) = 0;
  /// @brief Set the operation to the ABS2 Callback.
  /// @param operation Operation string
  /// @details Set the operation to the ABS2 Callback.
  void set(const std::string &operation);
  /// @brief Set the operation and operand to the ABS2 Callback.
  /// @param operation Operation string
  /// @param operand Operand string
  /// @details Set the operation and operand to the ABS2 Callback.
  void set(const std::string &operation, const std::string &operand);
  /// @brief Provide a hint solution to the ABS2 solver.
  /// @param hint Hint solution
  /// @details Provide a hint solution will be used by the ABS2 solver to
  /// improve the solution.
  void set(const Sol &hint);
  /// @brief Get the current solution.
  const Sol &get() const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace abs2

#endif  // ABS2_HPP_