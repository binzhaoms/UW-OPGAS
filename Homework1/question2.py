result = 1000
for i in range(10000):
    result -= 0.1
print(f"1000 - 10000 * 0.1")
print(f"result: {result}")

result = 10000
for i in range(100000):
    result -= 0.1
print(f"10000 - 100000 * 0.1")
print(f"result: {result}")

result = 100000
for i in range(1000000):
    result -= 0.1
print(f"100000 - 1000000 * 0.1")
print(f"result: {result}")