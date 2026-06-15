n = 111
sum = 0
checksum = 0

for i in range(n):
    sum += i + 8

for j in range(n):
    checksum += (j + 8) * 5

print(sum, checksum)
