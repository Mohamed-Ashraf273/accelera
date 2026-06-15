n = 118
sum = 0
checksum = 0

for i in range(n):
    sum += i + 13

for j in range(n):
    checksum += (j + 13) * 3

print(sum, checksum)
