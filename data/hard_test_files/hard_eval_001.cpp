#include <iostream>
#include <vector>

int main() {
  const int rows = 1200;
  const int cols = 1200;
  std::vector<double> a(rows * cols);
  std::vector<double> b(rows * cols);
  std::vector<double> c(rows * cols);
  double checksum = 0.0;

  for (int i = 0; i < rows * cols; i++) {
    a[i] = (i % 97) * 0.25;
    b[i] = (i % 53) * 0.5;
  }

  for (int r = 0; r < rows; r++) {
    for (int col = 0; col < cols; col++) {
      int idx = r * cols + col;
      double value = a[idx] + b[idx];
      for (int k = 0; k < 40; k++) {
        value = value * 1.000001 + 0.000001 * k;
      }
      c[idx] = value;
    }
  }

  for (int i = 0; i < rows * cols; i++) {
    checksum += c[i];
  }

  std::cout << checksum << std::endl;
  return 0;
}
