# Terms and Conditions
* The ABS2 QUBO Solver can be utilized free of charge only for non-commercial use, evaluation, and research purposes with no warranties.
* &copy; 2024 Koji Nakano. All rights reserved.

# ABS2 QUBO Solver GPU Engine C++ API
* The ABS2 C++ API provides the functionality of ABS2 QUBO Solver GPU-Engine.
* Namespace `abs2` is used.

# QUBO  problem
**A QUBO (Quadratic Unconstrained Optimization Problem) model** is a quadratic formula of binary variables.
The ABS2 QUBO Solver assumes that it has $\it{size}$ binary (0/1) variables $x_0, x_1, \ldots, x_{\it{size}-1}$.
We represent the integer coefficients of the quadratic terms $x_ix_j$ as $W_{i,j}$ ($0\leq i\leq j\leq n-1$). Consequently, a QUBO model can be expressed using an $size\times size$ integer upper triangular matrix $W=(W_{i,j})$ ($0\leq i\leq j\leq n-1$). This matrix defines the QUBO model $E(X)$ as follows:
$$
E(X)=\sum_{i=0}^{size-1}\sum_{i=j}^{size-1}W_{i,j}x_ix_j.
$$
The value of $E(X)$ is referred to as **the energy** of $X$. **The QUBO problem** aims to discover the optimal solution $X$ for a given QUBO model, minimizing the value of the energy $E(X)$.

# Outline of the ABS 2 QUBO Solver GPU-Engine
* Designed for a Linux server equipped with one or more GPUs.
* Based on the hybrid of GA (Genetic Algorithm) and SA (Simulated Annealing).
* The host Linux server manages multiple solution pools, each storing multiple solutions.
* The GPUs execute SA-based local search algorithms to discover solutions.
* For details, refer to [Related Publications](#related-publications).

# Outline of the ABS2 API
The ABS2 C++ API comprises classes: `abs2::Solver`, `abs2::Model`, `abs2::Param`, `abs2::Sol`, and  `abs2::Callback`.
* `abs2::Solver`: A C++ class to craate and initialize the ABS2 QUBO Solver.
* `abs2::Model`: A C++ class used to store and manipulate a QUBO model $W$ and execute the ABS2 QUBO solver.
  
* `abs2::Param`: A C++ class created to store various parameters for solving QUBO problems using the ABS2.
* `abs2::Sol`: A C++ class to store a solution $X$ for a QUBO problem.
* `abs2::Callback`: A C++ class for a callback function for the detailed control of the ABS QUBO solver. Users should define an inherited class from this class, overriding the callback function

The following files are provided for utilizing ABS2 QUBO solvers through the C++ API:
* `abs2.hpp`: A C++ header file, which must be included in C++ programs that uses the ABS2 C++ API.
* `libabs2.so`: A shared library for the ABS2 C++ API.
* `qubo*_*_*.so`: These files comprise the ABS2 GPU engine complied for a specific GPU architecture and should reside within the identical directory as `libabs2.so`.

# Installation
* Extract the contents of the tar.gz archive.
* The archive consists of directories:  `abs2/include`, `abs2/lib`, `abs2/doc`, `abs2/sample` and `abs2/csample`
* `abs2/include` contains `abs2.hpp` and `abs2c.h` .
* `abs2/lib`  contains `libabs2.so` (for ABS2 C++ API), `libabs2c.so` (for ABS2 C API) and shared libraries for the GPU-engine.
  * `libabs2.so`, `libabs2c.so`, and other shared libraries `qubo*_*_*.so` must be in the same directory.
* `abs2/sample` includes sample C++ programs that utilize the ABS2 C++ API.
* `abs2/csample` includes sample C programs that utilize the ABS2 C API.
* To successfully compile and execute, ensure the following environment variables are appropriately set:
  * `LIBRARY_PATH` and `LD_LIBRARY_PATH` should include `abs2/lib`.
  * `CPLUS_INCLUDE_PATH` (for ABS2 C++ API) and `C_INCLUDE_PATH` (for ABS2 C API) should include `abs2/include`.

# class `abs2::Solver`
* This classes is used to create and initialize ABS2 QUBO solver, and returns the various solver information.
* `abs2::Solver()`: A constructor to create and initialize ABS2 QUBO solver.
* `std::string abs2::get(const std::string &key)`: Returns the attribute value of `key`.
* `abs2::Solver operator()(const Model &model, const Param &param) const` : Executes the ABS2 QUBO solver for `model` with `param` and returns the obtained solution.
* `abs2::Solver operator()(const ABS2Model &model, const abs2::Sol &start) const`: Executes the ABS2 QUBO solver  for `model` with `param` and start solution `start` and returns the obtained solution.
* `void abs2::Solver::print(std::ostream &output_stream) const`: Prints the Solver attributes in the stream specified by `output_stream`.

## Attribute keys of `abs2::Solver`
* `solver`: solver name
* `version`: solver version
* `license`: license information
* `max_size`: supported maximum QUBO size
* `max_bits`: supported maximum coefficient bits 


# Class `abs2::Model`
* This class facilitates the storage and manipulation of a QUBO model alongside the execution of the ABS2 QUBO solver.
* `abs2::Model(int size, int bits)`: A constructor to generate a zero-initialized QUBO model with `size` variables and `bits`-bit coefficients. 
  * `size` must be from 32 to 65536.
  * `bits` must be either 8, 16, 32, or 64.
  * All coefficients $W_{i,j}$ ($0\leq i\leq j\leq size-1$) are zero-initialized.
* `abs2::Model(const std::string &filename)`: A constructor to load a QUBO model from the specified `filename`.
  *  The file must follow [Model file format](#model-file-format).
* `void abs2::Model::set(int i,int j, int64_t val)`: Sets the value `val` to $W_{i,j}$ if $i\leq j$ and to $W_{j,i}$ if $i>j$.
* `int64_t abs2::Model::get(int i,int j)` :  Returns the value of $W_{i,j}$ if $i\leq j$ and $W_{j,i}$ if $i>j$.
* `void abs2::Model::print(std::ostream &output_stream) const`: Prints the QUBO model in the stream specified by `output_stream`.

# Class `abs2::Param`
* This class stores parameters to be passed to the ABS2 QUBO Solver.
* `abs2::Param()`: A constructor to create an empty class instance to store parameters.
* `void abs2::Param::set(const std::string &key, const std::string &val)`: Assigns a parameter (`key`, `val`) where `key` represents the parameter name, and `val` consists of one or more decimal numbers separated by white space.
* `void set(Callback &callback)`: Sets an inheritance class of `abs2::Callback` with a user-defined overrided function `callback`.
* `void abs2::Param::print(std::ostream &output) const`: Print all stored parameters in output.

## Parameter keys of `abs2::Param`
Some of these parameters have a substantial impact on the search performance of the ABS2 QUBO solver when tackling individual QUBO problems. Unfortunately, determining the optimal parameters for solving these specific QUBO problems remains elusive. Developing a sophisticated method to identify optimal parameters for a given QUBO problem poses a significant research challenge.

* `"time_limit"`: The ABS2 terminates when the running time exceeds `time_limit` seconds.
* `"arithmetic_bits"`: 32 or 64. The number of bits used by arithmetic operations performed by the ABS2 QUBO solver.
* `"target_energy"`: The ABS2 QUBO solver terminates when a solution with the energy equal to or better than `target_energy` is obtained.
* `"search_cycles"`: Specifies two integers `batch_cycles` and `annealing_cycles`. This setting orchestrates duration of a single **batch search** on the GPU, executing `batch_cycles` cycles that consists of **annealing** with `annealing_cycles` cycles each. A **cycle** is referred to as a 1-bit flip operation of the annealing.
* `"best_priority"`: Prioritizes search operations around the best solution. Must take an integer value from `"-10"` (low) to `"10"` (high). A higher priority setting may lead to a focus on refining the current best solution, potentially resulting in stacking at that point. Conversely, a lower priority setting may cause the algorithm to overlook an optimal solution, even if it is in close proximity to the current best solution.
* `"verbose"`: Set to `"1"` to display detailed outputs from the ABS2 QUBO solver.
* `"gpus"`: Specifies the number of GPUs in use, which should not exceed the number of available GPUs. If not specified, all available GPUs will be utilized.
* `"islands_per_gpu"`: Specifies the number of islands managed by each GPU.
* `"blocks_per_island"`: Specifies the number of CUDA blocks working for each island. Maximum possible CUDA blocks will be utilized if not specified.
* `"pool_size"`: Specifies the pool size, indicating the number of solutions stored in the solution pool for each island.
* `"restart_duration"`: Specifies two numbers $t_0$ and $b$. The ABS QUBO solver restarts every $t_0*b^i$ seconds in $i$-th ($=0,1,...$) iteration. In the default setting, $t_0=1$ and $b=2$. This setting is crucial to prevent the ABS QUBO solver from getting stuck in a non-optimal local minimum solution. This restart operation is inactive if $t_0=0$.
* `"restart_nobest"`: During $t_0 \cdot b^i \cdot \it{self}$ seconds in the $i$-th iteration of $t_0 \cdot b^i$ seconds, the ABS2 QUBO solver does not utilize the best solution obtained in previous iterations. This setting is important  for exploring solutions distant from the best-known solution.
* `"tabu_cycles"`: Must range from 0 to 255. Once a bit is flipped during the annealing, it will remain unchanged in the subsequent `tabu_cycles` cycles.
* `"solution_proximity"`: Specifies the Hamming distance between solutions eligible for storage within a solution pool. If the Hamming distance between two solutions is smaller than `solution_proximity, both solutions will not simultaneously inhabit the solution pool to ensure its diversity.

# Class `abs2::Sol`
* A class to store a solution $X=(x_i)$ of a QUBO problem.
* `abs2::Sol(int size)` : A constructor to store a zero-initialized solution for QUBO problems of `size`. Since `size` is automatically adjusted upon storing a solution, `size` can be 0 if its value is unknown.
* `void abs2::Sol::set(int i, bool val)`: Sets `val` to $x_i$.
* `bool abs2::Sol::get(int i) const` : Returns the value of $x_i$.
* `std::string abs2::Sol::get(const std::string &key) const`: Returns `val` of attribute `key`. Both `key` and `val` are strings, even if they are numbers. Appropriate conversions using `std::stoi` or `std::stod` may be used for numbers.
* `void print(const std::string &option = "",std::ostream &output_stream = std::cout) const`: Print the solution, attributes, and logs to output_stream.
  * `option` is omitted: Prints the solution, all attributes, envs and logs.
  * `option="sol"`: Prints the solution.
  * `option="attrs"`: Prints all attributes.
  * `option="env"`: Prints all environment values applied to the ABS2 solver.
  * `option="log"`: Prints the log, a list of pairs the tts and the energy of newly obtained solutions.
  *  A character used as a separator can be specified by appending it to `option`. For example, `option="attrs,"` prints attributes separated by `","`, and `option="attrs\n"` prints them separated by line breaks.

## Solution attribute keys and values of `abs2::Sol`
* `"solver"`: The solver name.
* `"energy"`: The energy of the solution.
* `"tts"`: The Time To Solution in seconds to obtain the solution.
* `"runtime`: The running time of the ABS2 QUBO solver in seconds.
* `"kernel_time`: The running time of the GPU kernel.

# Class `abs2::Callback`
* Users can utilize a callback function by defining an inherited class from `abs2::Callback`, implementing a function override for `callback`.
* The ABS2 QUBO solver triggers a user-defined callback function, `callback`, upon the occurrence of specific events. These events can be categorized as `"init"`, `"new"`, or `"alarm"`.
* `"init"`: The callback function is precisely called once with the event `"init"` as soon as the ABS2 QUBO solver starts searching for solutions.
* `"new"`: The callback function is invoked with the event `"new"` whenever it discovers a new solution superior to the previous one.
* `"alarm"`: The callback function will be invoked after the duration of alarm duration.
* `virtual void abs2::Callback::callback(const std::string &event)`: This function should be defined by users in their inherited class of `abs2::Callback`. The `event` argument stores one of these events.
`void abs2::Callback::set(const std::string &operation)`: This function allows the setting of operations. It can be called during the execution of the callback.
  * `set("new")`: Enables the `"new"` event callback.
  * `set("abort")`: Terminates the ABS2 QUBO solver once all ongoing and scheduled searches on the GPUs are completed.
  * `set("kill")`: Immediately halts the ABS2 QUBO solver. All ongoing searches are terminated instantly and solutions obtained from them are discarded. Additionally, scheduled searches are canceled.
  * `set("restart")`: Discards acquired solutions and restarts the ABS2 QUBO solver. This can be executed if the ABS2 QUBO solver is trapped in a non-optimal local minimum solution and is not expected to escape from it.
* `void abs2::Callback::set(const std::string &operation, const std::string &operand)`: This function allows the setting of operations with operands. It can be called during the execution of the callback.
  * `set("new",operand)`: Enables the `"new"` event callback if the operand is `"1"` and disables it if the operand is `"0"`.
  * `set("alarm",operand)`: Enables the `"alarm"` event callback, triggering it after the specified number of `operand` seconds. Once the callback function returns, the callback will be invoked after the specified duration. Note that `"alarm"` is automatically cleared, and mut be set again for repeated alarm operation. Also, note that `operand` must be a decimal string.
  * `set("search_cycles",operand)`: Updates parameter `"search_cycles"`  by `operand`.
* `void abs2::Callback::set(const abs2::Sol &hint)`: Provides a potent solution `hint` to the ABS2.


# File formats
## Model file format
* A plain text storing decimal integer numbers.
* The fist line must contain four decimal integer numbers below separated by a white space.
  * `size`: the size (or the bit count) of QUBO problem.
  * `coefficients`: the number of non-zero coefficients in $W=(w_{i,j})$.
  * `min`: the value of $\min(W)$
  * `max`: the value of $\max(W)$
* `coefficients` lines follows. Each line must contain three numbers $i$, $j$, and $W_{i,j}$ ($\neq 0$) separated by white spaces.
* Following lines can contain any plain text.

## Solution file format
* A plain text storing binary (0/1) numbers.
* The first line must contain a bit vector of solution $x_0x_1\cdots x_{size-1}$ using ASCII characters "0" and "1", which should not be separated by white spaces.
* Following lines can contain any plain text.

# Sample Programs
### Solving a randomly generated QUBO problem.
A QUBO problem of size 32 is randomly generated and execute the ABS2 QUBO solver in 10 seconds.
```C++
#include "abs2.hpp"

int main(int argc, char *argv[]) {
  // Create QUBO "model" of 32 binary variables and 8-bit coefficients.
  abs2::Model model(32, 8);
  // Set a random integer from -10 to 10 to each W_{i,j}.
  for (int i = 0; i < 32; ++i)
    for (int j = i; j < 32; ++j)
      model.set(i, j, rand() % 21 - 10);
  // print the generated QUBO model
  model.print();
  // Create "param" to store parameters.
  abs2::Param param;
  // Set "time_limit" to 10 seconds.
  param.set("time_limit", "10");
  // Create and initializes the ABS2 QUBO solver
  abs2::Solver solver;
  // Solve "model" with "param" and store the solution in "sol".
  auto sol = solver(model, param);
  // Print the solution "sol".
  sol.print();
}
```

### Solving a QUBO problem stored in a file.
The QUBO problem stored in 'cat.mm.gz' is loaded and executed using the ABS2 QUBO solver within a time limit of 100 seconds.
Additionally, a callback function is set up to display all newly obtained best solutions.

```C++
#include "abs2.hpp"

class MyABS2Callback : public abs2::Callback {
  void callback(const std::string &event) {
    if (event == "init") {
      // Enable the callback for every new best solution.
      set("new");
    } else if (event == "new") {
      // Get the new best solution.
      auto sol = get();
      // Print the attributes of the new best solution.
      sol.print("attrs");
    }
  }
};

int main(int argc, char *argv[]) {
  // Create ABS2 Solver
  abs2::Solver solver;
  // Load QUBO model from the file "cat.mm.gz".
  abs2::Model model("../data/cat.mm.gz");
  // Create "param" to store parameters.
  abs2::Param param;
  // Set "time_limit" to 100 seconds.
  param.set("time_limit", "100");
  // Set apporiate search parameters for "cat.mm.gz".
  param.set("search_cycles", "500 100");
  // Never restart
  param.set("restart_duration", "0 0");
  param.set("best_priority", "10");
  // Set callback function.
  MyABS2Callback callback;
  param.set(callback);
  // Solve "model" with "param" and store the solution in "sol".
  auto sol = solver(model, param);
  // Print the solution "sol".
  sol.print();
}
```


# Related publications
* Koji Nakano, Daisuke Takafuji, Yasuaki Ito, Takashi Yazane, Junko Yano, Shiro Ozaki, Ryota Katsuki, Rie Mori: Diverse Adaptive Bulk Search: a Framework for Solving QUBO Problems on Multiple GPUs. IPDPS Workshops 2023: 314-325 (https://doi.org/10.1109/ipdpsw59300.2023.00060)
* Ryota Yasudo, Koji Nakano, Yasuaki Ito, Masaru Tatekawa, Ryota Katsuki, Takashi Yazane, Yoko Inaba: Adaptive Bulk Search: Solving Quadratic Unconstrained Binary Optimization Problems on Multiple GPUs. ICPP 2020: 62:1-62:11 (https://doi.org/10.1145/3404397.3404423)

# Appendix: A simple C wrapper for the ABS2 C++ API
For users who need to work with the C programming language, we have developed a straightforward C wrapper for the ABS2 C++ API.
Please be aware that this C wrapper does not encompass the complete functionality of the ABS2 C++ API.
The following files are provided for the C wrapper.
* `abs2c.h`: The header for the C wrapper. This file must be in the search path specified `C_INCLUDE_PATH`.
* `libabs2c.so`: The shared library for the C wrapper.
  
This should help bridge the gap for those using C in conjunction with ABS2, although note that the functionality might not be as extensive as the original C++ API.


## C wrapper data types
C wrapper data types defined in `abs2c.h`
* `abs2SolverPtr`: a pointer to a QUBO solver, instance of `abs2::Solver`.
* `abs2ModelPtr`: a pointer to a QUBO model data structure, instance of `abs2::Model`.
* `abs2ParamPtr`: a pointer to parameters given to the ABS2 QUBO solver, instance of `abs2::Param`
* `abs2SolPtr`: a pointer to a solution of the QUBO problem, instance of `abs2::Sol`
* `abs2CallbackPtr`: a pointer to the callback function, instance of `abs2::Callback`

## C wrapper functions
C API functions defined in `abs2c.h`

### Solver related functions
* `abs2SolverPtr abs2Solver_gen()`: Generates and initializes the ABS QUBO solver.
* `const char *abs2Solver_get_attr(abs2SolverPtr solver, const char *key)`: Returns the value of `key` of `solver`.

### Model related functions
* `abs2ModelPtr abs2Model_gen(int size, int bits)`: Generates an empty QUBO model of `size` variables each $W_{i,j}$ having `bits` bits.
* `abs2ModelPtr abs2Model_load(const char *filename)`: Load a QUBO model from file `filename`.
* `void abs2Model_del(abs2ModelPtr model)`: Deletes `model`.
* `void abs2Model_set(abs2ModelPtr model, int i, int j, int64_t val)`: Sets `val` to $W_{i,j}$ in `model`.
* `int64_t abs2Model_get(abs2ModelPtr model, int i, int j)`: Gets val of $W_{i,j}$ in `model`.
* `void abs2Model_set_attr(abs2ModelPtr model, const char *key, const char *val)`: Sets `val` to attribute `key` in `model`.
* `const char *abs2Model_get_attr(abs2ModelPtr model, const char *key)`: Gets the value of `key` in `model`.
* `void abs2Model_print(abs2ModelPtr model)`: Prints `model` in stdout.

### Parameter related functions
* `abs2ParamPtr abs2Param_gen()`: Generates an empty data for storing parameters.
* `abs2ParamPtr abs2Param_del(abs2ParamPtr param)`: Deletes `param`
* `abs2ParamPtr abs2Param_set(abs2ParamPtr param, const char *key, const char *val)`: Sets `val` to parameter `key` in `model`.
* `const char *abs2Param_get(abs2ParamPtr param, const char *key)`: Gets the value of `key`.
* `void abs2Param_print(abs2ParamPtr param)`: Prints `param` to stdout.

### Solution related functions
* `abs2SolPtr abs2Sol_gen(int size)`: Allocates memory for `sol` of `size` bits and returns the pointer to it. If size is unknown, `size` can be 0.
* `void abs2Sol_del(abs2SolPtr sol)`: Deletes `sol`.
* `void abs2Sol_set(abs2SolPtr sol, int i, bool val)`: Sets `val` to $x_i$ in `sol`.
* `bool abs2Sol_get(abs2SolPtr sol, int i)`: Gets the val of $x_i$ in `sol`.
* `void sbs2Sol_set_attr(abs2SolPtr sol, const char *key, const char *val)`: Sets `val` to `key`.
* `const char *abs2Sol_get_attr(abs2SolPtr sol, const char *key)`: Gets the value of `key`.
* `void abs2Sol_print(abs2SolPtr sol)`: Prints `sol` to stdout.

### Invoking ABS2 QUBO solver.
* `abs2Solve(abs2SolverPtr solver, abs2SolPtr sol, abs2ModelPtr model, abs2ParamPtr param)`: Solves `model` with `param` and store the solution in `sol` using `solver`
* `void abs2Solve_start(abs2SolverPtr solver, abs2SolPtr sol, abs2ModelPtr model, abs2ParamPtr param, abs2SolPtr start)`:
Solves `model` with `param` and `start` and stores the solution in `sol` using `solver`.

### Callback related functions
* `abs2CallbackPtr abs2param_callback(abs2ParamPtr param, void (*callback)(const char *event))`: Sets callback function `callback` to `param` and returns a pointer to store the callback function data. This C funciton is a wrapper of the C++ member function `void abs2::Callback::set(const std::string &operation)`.
* `void abs2callback_set(abs2CallbackPtr cb, const char *operation)`: Sets `operation` to `cb`. This C function is a wrapper of the C++ member function `void abs2::Callback::set(const std::string &operation, const std::string &operand)`.
* `void abs2callback_set_operand(abs2CallbackPtr cb, const char *operation, const char *operand)`: Sets `operation` with `operand` to `cb`.
* `void abs2callback_set_hint(abs2CallbackPtr cb, abs2SolPtr hint)`: Sets `hint` to `cb`. * `abs2SolPtr abs2callback_get(abs2CallbackPtr cb)`: Returns the current solution.

## Sample programs using the C wrapper.
### Solving a randomly generated QUBO problem.
```C
#include "abs2c.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // Create ABS2 QUBO Solver
  abs2SolverPtr solver = abs2Solver_gen();
  // Create QUBO "model" of 32 binary variables and 8-bit coefficients.
  abs2ModelPtr model = abs2Model_gen(32, 8);
  // Set a random integer from -10 to 10 to each W_{i,j}.
  for (int i = 0; i < 32; ++i)
    for (int j = i; j < 32; ++j)
      abs2Model_set(model, i, j, rand() % 21 - 10);
  // print the generated QUBO model
  abs2Model_print(model);
  // Create "param" to store parameters.
  abs2ParamPtr param = abs2Param_gen();
  // Set "time_limit" to 10 seconds.
  abs2Param_set(param, "time_limit", "10");
  abs2SolPtr sol = abs2Sol_gen(0);
  // Solve "model" with "param" and store the solution in "sol".
  abs2Solve(solver, sol, model, param);
  // Print the solution "sol".
  abs2Sol_print(sol);
}
```

### Solving a QUBO problem stored in a file.
```C
#include "abs2c.h"
#include "string.h"

abs2CallbackPtr cb;

void mycallback(const char *event) {
  if (strcmp(event, "init") == 0) {
    // set callback option to call callback when new solution is found.
    abs2Callback_set(cb, "new");
  } else if (strcmp(event, "new") == 0) {
    // print solution if new solution is found
    abs2SolPtr sol = abs2Callback_get(cb);
    abs2Sol_print(sol);
  }
}

int main(int argc, char *argv[]) {
  // Create ABS2 QUBO Solver
  abs2SolverPtr solver = abs2Solver_gen();
  // Load QUBO model from the file "cat.mm.gz".
  abs2ModelPtr model = abs2Model_load("../data/cat.mm.gz");
  // Create "param" to store parameters.
  abs2ParamPtr param = abs2Param_gen();
  // Set "time_limit" to 100 seconds.
  abs2Param_set(param, "time_limit", "100");
  // Set appropriate search parameters for "cat.mm.gz".
  abs2Param_set(param, "search_cycles", "500 100");
  abs2Param_set(param, "best_priority", "10");
  // Set callback function mycallback to cb
  cb = abs2Param_callback(param, mycallback);
  // Create "sol" to store the solution.
  abs2SolPtr sol = abs2Sol_gen(0);
  // Solve "model" with "param" and store the solution in "sol".
  abs2Solve(solver, sol, model, param);
  // Print the solution "sol".
  abs2Sol_print(sol);
}
```

# Release Notes

## Ver: 2024-01-21
* Initial release.

## Version: 2024-03-19
### C++ API Updates:
* Removed the `abs2::Init()` function.
* Introduced the `abs2::Solver` class. An `abs2::Solver` instance is now required to run the ABS2 QUBO solver.
* Eliminated the ABS2 QUBO solver execution functions from the `abs2::Model` class. Please use the `abs2::Solver` class function instead.

### C API Updates:
* Removed the `abs2Init()` function.
* Use `abs2solver_gen()` to initialize the ABS2 QUBO solver.
* The `abs2Solve()` function now requires a solver instance created by `abs2solver_gen()` to be specified.

## Version: 2024-05-01
* Namespace `ABS2` is changed to `abs2`.
