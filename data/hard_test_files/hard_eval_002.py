n = 50000000
total = 0
weighted = 0

for i in range(n):
    total += i % 100

for j in range(n):
    weighted += (j % 100) * 3

print(total, weighted)
