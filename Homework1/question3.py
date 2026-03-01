import numpy as np
from IPython.display import display

A = np.matrix([
    [1,2],
    [-1,1]
])

B = np.matrix([
    [2,0],
    [0,2]
])

C = np.matrix([
    [2,0,-3],
    [0,0,-1]
])

D = np.matrix([
    [1,2],
    [2,3],
    [-1,0]
])

x = np.matrix([
    [1],
    [0]
])

y = np.matrix([
    [0],
    [1]
])

z = np.matrix([
    [1],
    [2],
    [-1]
])

# (a) A + B
result = A + B
print("(a) A + B")
print(result)

# (b) 3x- 4y
result = 3*x - 4*y
print("(b) 3x- 4y")
print(result)

# (c) Ax
result = A*x
print("(c) Ax")
print(result)

# (d) B(x - y)
result = B*(x - y)
print("(d) B(x - y)")
print(result)

# (e) D x
result = D*x
print("(e) D x")
print(result)

# (f) D y + z
result = D*y + z
print("(f) D y + z")
print(result)

# (g) AB
result = A*B
print("(g) AB")
print(result)

# (h) BC
result = B*C
print("(h) BC")
print(result)

# (i) CD
result = C*D
print("(i) CD")
print(result)