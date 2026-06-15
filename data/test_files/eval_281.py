n = 62
sum = 0
checksum = 0

for i in range(n):
    sum += i + 9

for j in range(n):
    checksum += (j + 9) * 3

print(sum, checksum)
