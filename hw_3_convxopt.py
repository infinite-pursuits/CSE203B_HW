import cvxpy as cp
import numpy as np
import matplotlib. pyplot as plt

with open('/Users/chhaviyadav/Downloads/data_noise.txt') as f:
    lines = f.readlines()
    y_n = np.asarray([float(l.strip().split()[0]) for l in lines]).reshape((len(lines),1))

t = np.linspace(0, 5, num=5001)
f = np.linspace(1, 100, num=100)
phi = np.array([[np.sin(2*np.pi*fk*ti) for fk in f] for ti in t])

alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
residuals = []
xs = []
probvals = []

for alpha in alphas:
    print(alpha)
    x=cp.Variable((100,1))
    obj = cp.Minimize((alpha*cp.norm(x,1))+cp.sum_squares(y_n-cp.matmul(phi,x)))
    prob = cp.Problem(obj)
    prob.solve()
    residuals.append(cp.norm(y_n - (phi @ x)).value)
    xs.append(x.value)
    probvals.append(prob.value)

print(residuals)
print(probvals)

plt.plot(alphas,probvals)
plt.xscale('log')
plt.show()

for i in range(len(xs)):
    plt.plot(range(1,101),xs[i])
    plt.show()
'''
plt.plot(range(1,101),xs[0])
plt.show()'''
