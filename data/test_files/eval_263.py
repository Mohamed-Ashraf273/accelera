n = 133
sum = 0
checksum = 0

for i in range(n):
    sum += i + 4

for j in range(n):
    checksum += (j + 4) * 6

print(sum, checksum)
