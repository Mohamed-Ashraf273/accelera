#include <iostream>

int main() {
    int n = 77;
    int sum = 0;
    int checksum = 0;

#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++) {
        sum += i + 13;
    }

#pragma omp parallel for reduction(+ : checksum)
    for (int j = 0; j < n; j++) {
        checksum += (j + 13) * 6;
    }

    std::cout << sum << " " << checksum << std::endl;
    return 0;
}
