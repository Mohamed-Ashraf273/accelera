n = 80
sum = 0
checksum = 0

for i in range(n):
    sum += i + 5

for j in range(n):
    checksum += (j + 5) * 4

print(sum, checksum)
