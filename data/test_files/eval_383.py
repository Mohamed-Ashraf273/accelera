n = 75
sum = 0
checksum = 0

for i in range(n):
    sum += i + 7

for j in range(n):
    checksum += (j + 7) * 7

print(sum, checksum)
