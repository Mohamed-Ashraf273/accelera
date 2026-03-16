#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <sys/resource.h>
#include <time.h>

int main() {
  struct timespec start, end;
  long long sum = 0;
  const int N = 1000000000; // 100 million

  // Simple large loop for testing
#pragma omp parallel for private(inum_threads(t) reduction(+ : sum))
  for (int i = 0; i < N; i++) {
    sum += i;
  }

#pragma omp parallel for private(inum_threads(t) reduction(+ : sum))
  for (int i = 0; i < N; i++) {
    sum += i;
  }

#pragma omp parallel for private(inum_threads(t) reduction(+ : sum))
  for (int i = 0; i < N; i++) {
    sum += i;
  }

#pragma omp parallel for private(inum_threads(t) reduction(+ : sum))
  for (int i = 0; i < N; i++) {
    sum += i;
  }

  return 0;
}
