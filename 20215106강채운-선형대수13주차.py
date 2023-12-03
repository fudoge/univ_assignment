
# matplotlib color code: color='#eeefff' or, r, g, b, k, y, m ,c, y

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import string
import turtle as t

# 3D figure 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # Axe3D object

# points a, b and, c
a1, a2, a3, a4, a5 = (0,0,1), (2,0,1), (2,1,1), (0,1,1), (0,0,1)
b1, b2, b3, b4, b5 = (1,1,1), (2,1,1), (2,-1,1), (1,-1,1), (1,1,1)

# matrix with row vectors of points
A = np.array([a1, a2, a3, a4, a5])
Anew=np.zeros((A.shape[0],A.shape[1]+1))
AT=np.zeros((A.shape[0],A.shape[1]))
Ac=np.ones((1,A.shape[0]))
Anew[:,:-1]=A
Anew[:,-1]=Ac


B = np.array([b1, b2, b3, b4, b5])
Bnew=np.zeros((B.shape[0],B.shape[1]+1))
BT=np.zeros((B.shape[0],B.shape[1]))
Bc=np.ones((1,B.shape[0]))
Bnew[:,:-1]=B
Bnew[:,-1]=Bc

ax.plot(A[:,0], A[:,1], A[:,2], color='k', alpha=0.6, marker='o')
ax.plot(B[:,0], B[:,1], B[:,2], color='k', alpha=0.6, marker='o')




# scaling transformation matrix
sx=1/2
sy=1/2
sz=1
T_s = np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])

AT_s = T_s @ Anew.T
AT=AT_s[:-1,:].T

ax.plot(AT[:,0], AT[:,1], AT[:,2], color='r',alpha=0.6, marker='o')

BT_s = T_s @ Bnew.T
BT=BT_s[:-1,:].T

ax.plot(BT[:,0], BT[:,1], BT[:,2], color='r',alpha=0.6, marker='o')




# overall scaling transformation matrix
sall=2
T_sall = np.array([[sall, 0, 0, 0], [0, sall, 0, 0], [0, 0, sall, 0], [0, 0, 0, 1]])

AT_a = T_sall @ Anew.T
ATa=AT_a[:-1,:].T

ax.plot(ATa[:,0], ATa[:,1], ATa[:,2], color='g',alpha=0.6, marker='o')

BT_a = T_sall @ Bnew.T
BTa=BT_a[:-1,:].T

ax.plot(BTa[:,0], BTa[:,1], BTa[:,2], color='g',alpha=0.6, marker='o')




# rotational in z
theta=np.pi*2/3
cs=np.cos(theta)
ss=np.sin(theta)

T_rz = np.array([[cs, ss, 0, 0], [-ss, cs, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
T_rx = np.array([[1, 0, 0, 0],[0, cs, ss, 0], [0, -ss, cs, 0], [0, 0, 0, 1]])
T_ry = np.array([[cs, 0, -ss, 0], [0, 1, 0, 0], [ss, 0, cs, 0], [0, 0, 0, 1]])

AT_rz = T_rz @ Anew.T
ATrz=AT_rz[:-1,:].T

BT_rz = T_rz @ Bnew.T
BTrz=BT_rz[:-1,:].T

ax.plot(ATrz[:,0], ATrz[:,1], ATrz[:,2], color='b',alpha=0.6, marker='o')
ax.plot(BTrz[:,0], BTrz[:,1], BTrz[:,2], color='b',alpha=0.6, marker='o')


# shearing
b= 0.85
c= 0.20
d= 0.75
f= 0.7
g= 0.5
i= 1
T_sh = np.array([[1, d, g, 0], [b, 1, i, 0], [c, f, 1, 0], [0, 0, 0, 1]])

AT_sh = T_sh @ Anew.T
ATsh=AT_sh[:-1,:].T

ax.plot(ATsh[:,0], ATsh[:,1], ATsh[:,2], color='c',alpha=0.6, marker='o')

BT_sh = T_sh @ Bnew.T
BTsh=BT_sh[:-1,:].T

ax.plot(BTsh[:,0], BTsh[:,1], BTsh[:,2], color='c',alpha=0.6, marker='o')


plt.show()

