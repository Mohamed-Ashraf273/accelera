#include <iostream>

int main() {
  int n = 107;
  int sum = 0;
  int checksum = 0;
  int values[107];

  for (int i = 0; i < n; i++) {
    values[i] = i + 2;
  }

  for (int i = 0; i < n; i++) {
    sum += values[i];
  }

  for (int i = 0; i < n; i++) {
    checksum += values[i] * 6;
  }

  std::cout << sum << " " << checksum << std::endl;
  return 0;
}
