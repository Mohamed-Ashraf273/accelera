n = 92
sum = 0
checksum = 0

for i in range(n):
    sum += i + 13

for j in range(n):
    checksum += (j + 13) * 5

print(sum, checksum)
