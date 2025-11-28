def add(*args):
    total = 0
    for num in args:
        total += num
    return total
print(add(5, 10))
print(add(20, 30, 20))
print(add(20, 30, 20,40))

