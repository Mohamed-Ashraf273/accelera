#include <iostream>

int main() {
  int n = 89;
  int sum = 0;
  int checksum = 0;
  int values[89];

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    values[i] = i + 10;
  }

#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < n; i++) {
    sum += values[i];
  }

#pragma omp parallel for reduction(+ : checksum)
  for (int i = 0; i < n; i++) {
    checksum += values[i] * 2;
  }

  std::cout << sum << " " << checksum << std::endl;
  return 0;
}
