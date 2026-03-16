#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <time.h>
#include <sys/resource.h>

int main() {
    struct timespec start, end;
    long long sum = 0;
    const int N = 1000000000; // 100 million

    // Simple large loop for testing
    for (int i = 0; i < N; i++) {
        sum += i;
    }

    for (int i = 0; i < N; i++) {
        sum += i;
    }

    for (int i = 0; i < N; i++) {
        sum += i;
    }

    for (int i = 0; i < N; i++) {
        sum += i;
    }

    return 0;
}
