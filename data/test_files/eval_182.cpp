#include <iostream>

int main() {
  int n = 52;
  int sum = 0;
  int checksum = 0;
  int values[52];

  for (int i = 0; i < n; i++) {
    values[i] = i + 1;
  }

  for (int i = 0; i < n; i++) {
    sum += values[i];
  }

  for (int i = 0; i < n; i++) {
    checksum += values[i] * 2;
  }

  std::cout << sum << " " << checksum << std::endl;
  return 0;
}
