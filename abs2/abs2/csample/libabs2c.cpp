#include <stdbool.h>
#include <stdint.h>

#include "abs2.hpp"

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

class C_abs2Callback : public abs2::Callback {
 public:
  void (*callback_func)(const char *event);
  void callback(const std::string &event) override {
    static std::string temp;
    temp = event;
    (*callback_func)(temp.c_str());
  }
};

extern "C" {

abs2SolverPtr abs2Solver_gen() {
  auto ptr = new abs2::Solver();
  return (abs2SolverPtr){ptr};
}

const char *abs2Solver_get_attr(abs2SolverPtr solver, const char *key) {
  static std::string temp;
  temp = ((abs2::Solver *)solver.ptr)->get(key);
  return temp.c_str();
}

void abs2Solver_print(abs2SolverPtr solver) {
  ((abs2::Solver *)solver.ptr)->print();
}

abs2ModelPtr abs2Model_gen(int size, int bits) {
  auto ptr = new abs2::Model(size, bits);
  return (abs2ModelPtr){ptr};
}

abs2ModelPtr abs2Model_load(const char *filename) {
  auto ptr = new abs2::Model(filename);
  return (abs2ModelPtr){ptr};
}

void abs2Model_del(abs2ModelPtr model) { delete (abs2::Model *)model.ptr; }

void abs2Model_set(abs2ModelPtr model, int i, int j, int64_t val) {
  ((abs2::Model *)model.ptr)->set(i, j, val);
}

int64_t abs2Model_get(abs2ModelPtr model, int i, int j) {
  return ((abs2::Model *)model.ptr)->get(i, j);
}

void abs2Model_set_attr(abs2ModelPtr model, const char *key, const char *val) {
  ((abs2::Model *)model.ptr)->set(key, val);
}

const char *abs2Model_get_attr(abs2ModelPtr model, const char *key) {
  static std::string temp;
  temp = (((abs2::Model *)model.ptr)->get(key)).c_str();
  return temp.c_str();
}

void abs2Model_print(abs2ModelPtr model) {
  ((abs2::Model *)model.ptr)->print();
}

abs2ParamPtr abs2Param_gen() {
  auto ptr = new abs2::Param();
  return (abs2ParamPtr){ptr};
}

void abs2Param_del(abs2ParamPtr param) { delete (abs2::Param *)param.ptr; }

void abs2Param_set(abs2ParamPtr param, const char *key, const char *val) {
  ((abs2::Param *)param.ptr)->set(key, val);
}

const char *abs2Param_get(abs2ParamPtr param, const char *key) {
  static std::string temp;
  temp = (((abs2::Param *)param.ptr)->get(key)).c_str();
  return temp.c_str();
}

abs2CallbackPtr abs2Param_callback(abs2ParamPtr param,
                                   void (*callback)(const char *event)) {
  static C_abs2Callback cb;
  cb.callback_func = callback;
  ((abs2::Param *)param.ptr)->set(cb);
  return (abs2CallbackPtr){&cb};
}

void abs2Callback_set(abs2CallbackPtr cb, const char *operation) {
  ((abs2::Callback *)cb.ptr)->set(operation);
}

void abs2Callback_set_operand(abs2CallbackPtr cb, const char *operation,
                              const char *operand) {
  ((abs2::Callback *)cb.ptr)->set(operation, operand);
}

void abs2Callback_set_hint(abs2CallbackPtr cb, abs2SolPtr hint) {
  ((abs2::Callback *)cb.ptr)->set(*((abs2::Sol *)hint.ptr));
}

abs2SolPtr abs2Callback_get(abs2CallbackPtr cb) {
  //  return (abs2SolPtr){&(((abs2::Callback *)cb.ptr)->get())};
  return (abs2SolPtr){const_cast<void *>(static_cast<const void *>(
      &(static_cast<const abs2::Callback *>(cb.ptr)->get())))};
}

void abs2Param_print(abs2ParamPtr param) {
  ((abs2::Param *)param.ptr)->print();
}

abs2SolPtr abs2Sol_gen(int size) {
  auto ptr = new abs2::Sol(size);
  return (abs2SolPtr){ptr};
}

void abs2Sol_del(abs2SolPtr sol) { delete (abs2::Sol *)sol.ptr; }

void abs2Sol_set(abs2SolPtr sol, int i, bool val) {
  ((abs2::Sol *)sol.ptr)->set(i, val);
}

bool abs2Sol_get(abs2SolPtr sol, int i) {
  return ((abs2::Sol *)sol.ptr)->get(i);
}

void sbs2Sol_set_attr(abs2SolPtr sol, const char *key, const char *val) {
  ((abs2::Sol *)sol.ptr)->set(key, val);
}

const char *abs2Sol_get_attr(abs2SolPtr sol, const char *key) {
  static std::string temp;
  temp = (((abs2::Sol *)sol.ptr)->get(key)).c_str();
  return temp.c_str();
}

void abs2Sol_print(abs2SolPtr sol) { ((abs2::Sol *)sol.ptr)->print(); }

void abs2Solve(abs2SolverPtr solver, abs2SolPtr sol, abs2ModelPtr model,
               abs2ParamPtr param) {
  abs2::Sol *abs2_sol = (abs2::Sol *)sol.ptr;
  const abs2::Solver *abs2_solver = (abs2::Solver *)solver.ptr;
  const abs2::Model *abs2_model = (abs2::Model *)model.ptr;
  const abs2::Param *abs2_param = (abs2::Param *)param.ptr;
  *abs2_sol = (*abs2_solver)(*abs2_model, *abs2_param);
}

void abs2Solve_start(abs2SolverPtr solver, abs2SolPtr sol, abs2ModelPtr model,
                     abs2ParamPtr param, abs2SolPtr start) {
  abs2::Sol *abs2_sol = (abs2::Sol *)sol.ptr;
  const abs2::Solver *abs2_solver = (abs2::Solver *)solver.ptr;
  const abs2::Model *abs2_model = (abs2::Model *)model.ptr;
  const abs2::Param *abs2_param = (abs2::Param *)param.ptr;
  abs2::Sol *abs2_start = (abs2::Sol *)start.ptr;
  *abs2_sol = (*abs2_solver)(*abs2_model, *abs2_param, *abs2_start);
}
}
