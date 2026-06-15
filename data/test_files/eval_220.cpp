#include <iostream>

int main() {
  int n = 90;
  int sum = 0;
  int checksum = 0;
  int values[90];

  for (int i = 0; i < n; i++) {
    values[i] = i + 13;
  }

  for (int i = 0; i < n; i++) {
    sum += values[i];
  }

  for (int i = 0; i < n; i++) {
    checksum += values[i] * 5;
  }

  std::cout << sum << " " << checksum << std::endl;
  return 0;
}
