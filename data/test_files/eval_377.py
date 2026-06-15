n = 69
sum = 0
checksum = 0

for i in range(n):
    sum += i + 1

for j in range(n):
    checksum += (j + 1) * 8

print(sum, checksum)
