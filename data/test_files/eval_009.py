n = 57
sum = 0
checksum = 0

for i in range(n):
    sum += i + 10

for j in range(n):
    checksum += (j + 10) * 4

print(sum, checksum)
