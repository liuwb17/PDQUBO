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