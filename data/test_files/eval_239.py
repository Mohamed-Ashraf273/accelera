n = 109
sum = 0
checksum = 0

for i in range(n):
    sum += i + 6

for j in range(n):
    checksum += (j + 6) * 3

print(sum, checksum)
