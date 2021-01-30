import time

s = 0.1

t = time.time()

i = 0
while i <= 500000:
    s += i
    i += 1
    print(s)

end = time.time()

print(end - t)
