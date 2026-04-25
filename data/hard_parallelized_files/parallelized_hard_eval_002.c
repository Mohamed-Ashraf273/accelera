#include <iostream>

int main() {
    int n = 30000000;
    int total = 0;
    int weighted = 0;

#pragma omp parallel for reduction(+ : total)
    for (int i = 0; i < n; i++) {
        total += i % 100;
    }

#pragma omp parallel for reduction(+ : weighted)
    for (int j = 0; j < n; j++) {
        weighted += (j % 100) * 3;
    }

    std::cout << total << " " << weighted << std::endl;
    return 0;
}
