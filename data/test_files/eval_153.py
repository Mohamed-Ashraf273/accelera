n = 112
sum = 0
checksum = 0

for i in range(n):
    sum += i + 11

for j in range(n):
    checksum += (j + 11) * 8

print(sum, checksum)
