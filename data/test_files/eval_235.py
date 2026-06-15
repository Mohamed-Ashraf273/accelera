n = 105
sum = 0
checksum = 0

for i in range(n):
    sum += i + 2

for j in range(n):
    checksum += (j + 2) * 6

print(sum, checksum)
