n = 59
sum = 0
checksum = 0

for i in range(n):
    sum += i + 12

for j in range(n):
    checksum += (j + 12) * 6

print(sum, checksum)
