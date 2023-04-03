import numpy as np

#1)    

def function(t, y):
    return (t - y**2)

def eulers(t, y, iters, i):
    j = (i - t) / iters

    for unused_variable in range(iters):
        y = y + (j * function(t, y))
        t = t + j
   
    print("%.5f" % y, "\n")

t = 0
y = 1
iters = 10
i = 2
eulers(t, y, iters, i)

#2)

def runge_kutta(t, y, iters, i):
    j = (i - t) / iters
   
    for unused_variable in range(iters):
        a = j * function(t, y)
        b = j * function((t + (j / 2)), (y + (a / 2)))
        c = j * function((t + (j / 2)), (y + (b / 2)))
        d = j * function((t + j), (y + c))

        y = y + (1 / 6) * (a + (2 * b) + (2 * c) + d)

        t = t + j

    print("%.5f" % y, "\n")

runge_kutta(t, y, iters, i)

#3)

def gaussian_elimination(matrix):
    size = matrix.shape[0]

    for i in range(size):
        pivot = i
        while matrix[pivot, i] == 0:
            pivot - pivot + 1
   
        matrix[[i, pivot]] = matrix[[pivot, i]]

        for j in range(i + 1, size):
            factor = matrix[j, i] / matrix[i, i]
            matrix[j, i:] = matrix[j, i:] - factor * matrix[i, i:]

    inputs = np.zeros(size)

    for i in range(size - 1, -1, -1):
        inputs[i] = (matrix[i, -1] - np.dot(matrix[i, i: -1], inputs[i:])) / matrix[i, i]
   
    aug_matrix = np.array([int(inputs[0]), int(inputs[1]), int(inputs[2])], dtype=np.double)
    print(aug_matrix, "\n")

gaussian_elimination(np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]]))

#4)

def lu_factor(matrix):
    size = matrix.shape[0]

    l = np.eye(size)
    u = np.zeros_like(matrix)

    for x in range(size):
        for y in range(x, size):
            u[x, y] = (matrix[x, y] - np.dot(l[x, :x], u[:x, y]))
   
        for y in range(x + 1, size):
            l[y, x] = (matrix[y, x] - np.dot(l[y, :x], u[:x, x])) / u[x, x]
   
    determinant = np.linalg.det(matrix)

    print("%.5f" % determinant, "\n")
    print(l, "\n")
    print(u, "\n")

lu_factor(np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype = np.double))

#5)

def diag_dom(matrix, num_rows):

    for x in range(0, num_rows):
        total = 0
        for y in range(0, num_rows):
            total = total + abs(matrix[x][y])
       
        total = total - abs(matrix[x][x])
   
    if abs(matrix[x][x]) < total:
        print("False\n")
    else:
        print("True\n")

num_rows = 5
matrix_5 = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
diag_dom(matrix_5, num_rows)

#6)

def pos_def(matrix):
    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        print("True")
    else:
        print("False")

matrix_6 = np.matrix([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
pos_def(matrix_6)