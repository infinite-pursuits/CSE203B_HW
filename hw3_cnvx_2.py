import cvxpy as cp
import numpy as np

c = [[1.4, 0.8, 1.3], [0.4, 1.0, 0.7], [0.9, 0.3, 0.6],
     [0.7, 1.2, 0.4], [1.4, 0.8, 1.1], [1.5, 0.1, 1.3]]

M = cp.Variable((3, 3), PSD=True)
y = cp.Variable(3)

obj = cp.Minimize(-cp.log_det(M))
constraints = [cp.sum_squares(M@ci - y) <= 1 for ci in c]

prob = cp.Problem(obj, constraints)
prob.solve()
print(M.value, y.value, prob.status)

R = M.value@M.value
z = np.linalg.inv(M.value)@y.value
print(R,z)