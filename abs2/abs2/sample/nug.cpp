#include "abs2.hpp"

class MyABS2Callback : public abs2::Callback {
  void callback(const std::string &event) {
    // Will callback in 2 secconds.
    set("alarm", "2");
    if (event == "init") {
      // Enable the callback for every new best solution.
      set("new");
    } else {
      auto sol = get();
      // Print the attributes of the new best solution.
      sol.print("attrs");
    }
  }
};

int main(int argc, char *argv[]) {
  // Create ABS2 QUBO solver
  abs2::Solver solver;
  // Load QUBO model from the file "nug30.mm.gz".
  abs2::Model model("../data/nug30.mm.gz");
  // Create "param" to store parameters.
  abs2::Param param;
  // Set "time_limit" to 10 seconds.
  param.set("time_limit", "10");
  // Solve the QUBO and store the solution in "hint".
  auto hint = solver(model, param);
  hint.print();
  // Set "time_limit" to 10 seconds.
  param.set("time_limit", "30");
  // terminates if the energy is -23876
  param.set("target_energy", "-23876");
  // Set callback function
  MyABS2Callback callback;
  param.set(callback);
  // Solve the QUBO with "hint" and store the solution in "sol".
  auto sol = solver(model, param, hint);
  // Print the solution "sol".
  sol.print();
}