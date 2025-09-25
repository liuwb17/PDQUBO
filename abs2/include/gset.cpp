#include <iostream>
#include <string>
#include <vector>

#include "abs2.hpp"
class MyABS2Callback : public abs2::Callback {
  void callback(const std::string& event) {
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

int main(int argc, char* argv[]) {
  int id = std::stoi(argv[1]);
  abs2::Solver solver;

  std::string instance_name = "../Gset_abs2/G" + std::to_string(id) + ".txt";

  abs2::Model model(instance_name);

  abs2::Param param;

  param.set("time_limit", "180");

  // Set callback function.
  MyABS2Callback callback;
  param.set(callback);
  // Solve "model" with "param" and store the solution in "sol".
  auto sol = solver(model, param);

  sol.print();
  return 0;
}