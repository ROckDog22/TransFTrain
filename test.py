shape = [3, 4, 5]
stride = 1
res = []
for i in range(1, len(shape) + 1):
    res.append(stride)
    stride *= shape[-i]
print(tuple(res[::-1]))