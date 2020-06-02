import numpy as np
class stack:
    def __init__(self, size=5):
        self.arr = np.zeros(size)
        self.size = size
        self.sum = 0
        self.i = 0

    def push(self, a):
        self.i = (self.i + 1) % self.size
        self.sum += -self.arr[self.i] + a
        self.arr[self.i] = a
        return self.sum

    def pop(self):
        return self.arr[self.i]

    def _print(self):
        print(self.arr, self.i)


class stackdif:
    def __init__(self, size=5):
        self.stack1 = stack(size)
        self.stack2 = stack(size)
        self.sum1 = self.stack1.sum
        self.sum2 = self.stack1.sum
        self.size = size
        self.j = 0
        self.full = False

    def push(self, a):
        self.j += 1
        if self.j > self.size * 2: self.full = True
        self.sum2 = self.stack2.push(self.stack1.pop())
        self.sum1 = self.stack1.push(a)

    def dif(self):
        if not self.full or self.sum2 == 0: return 0
        return (self.sum1 - self.sum2) / self.sum2

    def _print(self):
        self.stack1._print()
        self.stack2._print()