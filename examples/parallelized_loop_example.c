#include <iostream>

int main() {
    int sum = 1;
#pragma omp parallel for num_threads(t) reduction(+ : sum)
for (int i = 0; i < 5; i++) {
        sum += i;
    }
    return 0;
}
