#include "abs2.hpp"

int main(int argc, char *argv[]) {
  abs2::Solver solver(false);
  std::cout << "solver = " << solver.get("solver") << std::endl;
  std::cout << "version = " << solver.get("version") << std::endl;
  std::cout << "license = " << solver.get("license") << std::endl;
  std::cout << "max_size = " << solver.get("max_size") << std::endl;
  std::cout << "max_bits = " << solver.get("max_bits") << std::endl;
}