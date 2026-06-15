n = 128
sum = 0
checksum = 0

for i in range(n):
    sum += i + 1

for j in range(n):
    checksum += (j + 1) * 3

print(sum, checksum)
