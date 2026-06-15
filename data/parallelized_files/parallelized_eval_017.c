#include <iostream>

int main() {
    int n = 65;
    int sum = 0;
    int checksum = 0;

#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++) {
        sum += i + 5;
    }

#pragma omp parallel for reduction(+ : checksum)
    for (int j = 0; j < n; j++) {
        checksum += (j + 5) * 5;
    }

    std::cout << sum << " " << checksum << std::endl;
    return 0;
}
