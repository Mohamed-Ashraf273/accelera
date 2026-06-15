#include <iostream>

int main() {
  int n = 92;
  int sum = 0;
  int checksum = 0;
  int values[92];

  for (int i = 0; i < n; i++) {
    values[i] = i + 6;
  }

  for (int i = 0; i < n; i++) {
    sum += values[i];
  }

  for (int i = 0; i < n; i++) {
    checksum += values[i] * 4;
  }

  std::cout << sum << " " << checksum << std::endl;
  return 0;
}
