n = 53
sum = 0
checksum = 0

for i in range(n):
    sum += i + 2

for j in range(n):
    checksum += (j + 2) * 3

print(sum, checksum)
