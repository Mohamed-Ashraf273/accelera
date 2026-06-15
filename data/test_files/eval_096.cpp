#include <iostream>

int main() {
  int n = 55;
  int sum = 0;
  int checksum = 0;
  int values[55];

  for (int i = 0; i < n; i++) {
    values[i] = i + 6;
  }

  for (int i = 0; i < n; i++) {
    sum += values[i];
  }

  for (int i = 0; i < n; i++) {
    checksum += values[i] * 7;
  }

  std::cout << sum << " " << checksum << std::endl;
  return 0;
}
