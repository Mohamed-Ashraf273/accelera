n = 82
sum = 0
checksum = 0

for i in range(n):
    sum += i + 3

for j in range(n):
    checksum += (j + 3) * 2

print(sum, checksum)
