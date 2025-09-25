#ifndef ABS2C_H_
#define ABS2C_H_
#include <stdbool.h>
#include <stdint.h>

typedef struct abs2Solver {
  void *ptr;
} abs2SolverPtr;

typedef struct abs2Model {
  void *ptr;
} abs2ModelPtr;

typedef struct abs2Param {
  void *ptr;
} abs2ParamPtr;

typedef struct abs2Sol {
  void *ptr;
} abs2SolPtr;

typedef struct abs2Callback {
  void *ptr;
} abs2CallbackPtr;

// Generate and initializes ABS2 QUBO solver
abs2SolverPtr abs2Solver_gen();

// Get solver attr with "key".
const char *abs2Solver_get_attr(abs2SolverPtr solver, const char *key);

// Print solver attributes to stdout
void abs2Solver_print(abs2SolverPtr solver);

// Generate an empty QUBO model of "size" variables with each W_{i,j} has
// "bits" bits.
abs2ModelPtr abs2Model_gen(int size, int bits);

// Load QUBO model of mm foramt
abs2ModelPtr abs2Model_load(const char *filename);

// Delete "model"
void abs2Model_del(abs2ModelPtr model);

// Set "val" to W_{i,j}
void abs2Model_set(abs2ModelPtr model, int i, int j, int64_t val);

// Get val of W_{i,j}
int64_t abs2Model_get(abs2ModelPtr model, int i, int j);

// Get val of W_{i,j}
void abs2Model_set_attr(abs2ModelPtr model, const char *key, const char *val);

// Get the value of "key".
const char *abs2Model_get_attr(abs2ModelPtr model, const char *key);

// Print "model"  to stdout
void abs2Model_print(abs2ModelPtr model);

// Allocate memory for "param" with size and return the pointer to it.
abs2ParamPtr abs2Param_gen();

// Delete "param".
abs2ParamPtr abs2Param_del(abs2ParamPtr param);

// Set the value of "key".
void abs2Param_set(abs2ParamPtr param, const char *key, const char *val);

// Get the value of "key".
const char *abs2Param_get(abs2ParamPtr param, const char *key);

// Set callback
abs2CallbackPtr abs2Param_callback(abs2ParamPtr param,
                                   void (*callback)(const char *event));

void abs2Callback_set(abs2CallbackPtr cb, const char *operation);

void abs2Callback_set_operand(abs2CallbackPtr cb, const char *operation,
                              const char *operand);

void abs2Callback_set_hint(abs2CallbackPtr cb, abs2SolPtr hint);

abs2SolPtr abs2Callback_get(abs2CallbackPtr cb);

// Print param to stdout
void abs2Param_print(abs2ParamPtr param);

// Allocate memory for "sol" with size and return the pointer to it.
// size can be any.
abs2SolPtr abs2Sol_gen(int size);

// Delete "sol".
void abs2Sol_del(abs2SolPtr sol);

// Set the value to x_i.
void abs2Sol_set(abs2SolPtr sol, int i, bool val);

// Get the value of x_i
bool abs2Sol_get(abs2SolPtr sol, int i);

// Set solution attr with key.
void sbs2Sol_set_attr(abs2SolPtr sol, const char *key, const char *val);

// Get solution attr with "key".
const char *abs2Sol_get_attr(abs2SolPtr sol, const char *key);

// Print  "sol" to stdout
void abs2Sol_print(abs2SolPtr sol);

// Solve "model" with "param" and store the solution in "sol".
void abs2Solve(abs2SolverPtr solver, abs2SolPtr sol, abs2ModelPtr model,
               abs2ParamPtr param);

// Solve "model" with "param" and "start" and store the solution in "sol".
void abs2Solve_start(abs2SolverPtr solver, abs2SolPtr sol, abs2ModelPtr model,
                     abs2ParamPtr param, abs2SolPtr start);

#endif