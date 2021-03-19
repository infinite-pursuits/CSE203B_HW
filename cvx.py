import cvxpy as cp
import numpy as np

m=5
n=2

x=np.asarray([[0,0,0,0,0] ,[10,20,30,40,50]])
y=np.asarray([[-10,-20,-30,-40,-50],[0,0,0,0,0]])

lamda = cp.Variable(5)
mu = cp.Variable(5)
t_sum_x = cp.Variable(2)
t_sum_y = cp.Variable(2)
constraints = [cp.sum(lamda)==0.5, cp.sum(mu)==0.5,lamda>=0, mu>=0]


x = x.transpose()
y = y.transpose()

for i in range(5):
    t_sum_x = t_sum_x+(lamda[i]*x[i])
    t_sum_y = t_sum_y + (mu[i] * y[i])
obj = cp.Maximize(-cp.norm(t_sum_x-t_sum_y))

prob = cp.Problem(obj, constraints)
prob.solve()
v_x = np.zeros(2)
v_y = np.zeros(2)

for i in range(5):
    v_x = v_x+(lamda.value[i]*x[i])
    v_y = v_y + (mu.value[i] * y[i])

print("status:", prob.status)
print("optimal lambda", lamda.value)
print("optimal mu",mu.value)
print("max val", -1 * np.linalg.norm(v_x-v_y))
print("Hull point 1:",2*v_x)
print("Hull point 2:",2*v_y)

'''
o = np.ones((1,5))

u=cp.Variable()
b=cp.Variable()
a=cp.Variable((1,2))

constraints = [a@x<=(b-u)*o,a@y>=(b+u)*o,cp.sum_squares(a)<=1]
obj = cp.Maximize(u)
prob = cp.Problem(obj, constraints)
prob.solve()

print("status:", prob.status)
print("optimal u", u.value)
print("optimal b", b.value)
print("optimal a", a.value)
print(b.value-u.value)
print(b.value+u.value)
'''
