// Test file with various loops for extraction
#include <iostream>
#include <vector>

int main() {
  // Simple for loop
  for (int i = 0; i < 10; i++) {
    std::cout << i << std::endl;
  }

  // Nested for loops
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      std::cout << i * j << " ";
    }
    std::cout << std::endl;
  }

  // While loop
  int counter = 0;
  while (counter < 10) {
    std::cout << "Count: " << counter << std::endl;
    counter++;
  }

  // Range-based for loop
  std::vector<int> numbers = {1, 2, 3, 4, 5};
  for (const auto &num : numbers) {
    std::cout << num << std::endl;
  }

  // For loop with complex condition
  for (int i = 0; i < 100; i += 5) {
    if (i % 2 == 0) {
      std::cout << "Even: " << i << std::endl;
    }
  }

  return 0;
}
