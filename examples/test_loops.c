// Test file with various loops for extraction
#include <stdio.h>

int main() {
  // Simple for loop
  for (int i = 0; i < 10; i++) {
    printf("%d\n", i);
  }

  // Nested for loops
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%d ", i * j);
    }
    printf("\n");
  }

  // While loop
  int counter = 0;
  while (counter < 10) {
    printf("Count: %d\n", counter);
    counter++;
  }

  // Array iteration with for loop
  int numbers[] = {1, 2, 3, 4, 5};
  int numbers_size = sizeof(numbers) / sizeof(numbers[0]);
  for (int i = 0; i < numbers_size; i++) {
    printf("%d\n", numbers[i]);
  }

  // For loop with complex condition
  for (int i = 0; i < 100; i += 5) {
    if (i % 2 == 0) {
      printf("Even: %d\n", i);
    }
  }

  // Do-while loop
  int k = 0;
  do {
    printf("Do-while: %d\n", k);
    k++;
  } while (k < 5);

  return 0;
}
