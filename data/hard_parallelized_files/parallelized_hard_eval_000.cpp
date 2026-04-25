#include <cmath>
#include <iostream>
#include <vector>

int main() {
  const int n = 6000000;
  std::vector<double> input(n);
  std::vector<double> output(n);
  double checksum = 0.0;

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    input[i] = (i % 1000) * 0.001;
  }

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    double x = input[i];
    double value = x;
    for (int k = 0; k < 20; k++) {
      value = std::sin(value) + std::cos(x + k * 0.01);
    }
    output[i] = value;
  }

#pragma omp parallel for reduction(+ : checksum)
  for (int i = 0; i < n; i++) {
    checksum += output[i];
  }

  std::cout << checksum << std::endl;
  return 0;
}
