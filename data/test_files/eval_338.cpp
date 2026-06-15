#include <iostream>

int main() {
  int n = 119;
  int sum = 0;
  int checksum = 0;
  int values[119];

  for (int i = 0; i < n; i++) {
    values[i] = i + 1;
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
