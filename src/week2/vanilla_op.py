# 逐元素操作
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
c = []

for i in range(len(a)):
    c.append(a[i] + b[i])

print(c)  # [6, 8, 10, 12]