import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

import numpy as np
import string
import turtle as t

# 3D figure 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # Axe3D object
axisR = 20 # range of axis
xrange=[-axisR, axisR]
yrange=[-axisR, axisR]
zrange=[-axisR, axisR]
ax.set_xlim(xrange)
ax.set_ylim(yrange)
ax.set_zlim(zrange)

# define functions 
# function: make Anew from A; 
def ArrayNew(Array):
    Array_n = np.zeros((Array.shape[0],Array.shape[1]+1))
    AT_n=np.zeros((Array.shape[0],Array.shape[1]))
    Ac_n=np.ones((1,Array.shape[0]))
    Array_n[:,:-1]=Array
    Array_n[:,-1]=Ac_n
    return Array_n

# overall scaling transformation matrix: Array_n(before transformation) -> ArrayT (after transformation)
# T_s: transformation matrix with sx, sy, and sz as scaling factors in x, y, and z directions
def Tscale(sx,sy,sz, Array_n):
    T_s = np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])
    AT_s = T_s @ Array_n.T
    ArrayT=AT_s[:-1,:].T
    return ArrayT

# scaling transformation: Array_n(before transformation) -> ArrayT (after transformation)
# T_sall: transformation matrix with sall as a scaling factor in x, y, and z directions
def TscaleAll(sall, Array_n):
    T_sall = np.array([[sall, 0, 0, 0], [0, sall, 0, 0], [0, 0, sall, 0], [0, 0, 0, 1]])
    AT_s = T_sall @ Array_n.T
    ArrayT=AT_s[:-1,:].T
    return ArrayT

# rotation transformation: Array_n(before transformation) -> ATr (after transformation)
# T_rx: rotation in x
def Trx(theta,Attay_n): 
    theta=np.pi*2/3
    cs=np.cos(theta)
    ss=np.sin(theta)
    T_rx = np.array([[1, 0, 0, 0],[0, cs, ss, 0], [0, -ss, cs, 0], [0, 0, 0, 1]])
    AT_r = T_rx @ Attay_n.T
    ATr=AT_r[:-1,:].T
    return ATr

# rotation transformation: Array_n(before transformation) -> ATr (after transformation)
# T_rx: rotation in y
def Try(theta,Attay_n):
    theta=np.pi*2/3
    cs=np.cos(theta)
    ss=np.sin(theta)    
    T_ry = np.array([[cs, 0 -ss, 0], [0, 1, 0, 0], [ss, 0, cs, 0], [0, 0, 0, 1]])
    AT_r = T_ry @ Attay_n.T
    ATr=AT_r[:-1,:].T
    return ATr

# rotation transformation: Array_n(before transformation) -> ATr (after transformation)
# T_rx: rotation in z
def Trz(theta,Attay_n):
    theta=np.pi*2/3
    cs=np.cos(theta)
    ss=np.sin(theta)
    T_rz = np.array([[cs, ss, 0, 0], [-ss, cs, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])    
    AT_r = T_rz @ Attay_n.T
    ATr=AT_r[:-1,:].T
    return ATr

# translation transformation: Array_n(before transformation) -> ATtr (after transformation)
# tx, ty, and tz in x, y, and z direction
def Ttr(tx,ty,tz, Array_n):
    T_tr = np.array([[1, 0, 0, tx],[0, 1, 0, ty],[0, 0, 1, tz],[0,0,0,1]])
    AT_tr = T_tr @ Array_n.T
    ATtr = AT_tr[:-1,:].T
    return ATtr

# shearing transformation: Array_n(before transformation) -> ATsh (after transformation)
# T_sh: shearing in b,c,d,f,g,i
def Tsh(b,c,d,f,g,i,Attay_n): 
    T_sh = np.array([[1, d, g, 0], [b, 1, i, 0], [c, f, 1, 0], [0, 0, 0, 1]])    
    AT_sh = T_sh @ Attay_n.T
    ATsh=AT_sh[:-1,:].T
    return ATsh

# points of A rectangle of a hexahedron
def sixSideA(x,y,z):
    # points a, b and, c
    a1, a2, a3, a4, a5 = (0,0,z), (x,0,z), (x,y,z), (0,y,z), (0,0,z)
    A = np.array([a1, a2, a3, a4, a5])
    return A

# points of B rectangle of a hexahedron
def sixSideB(x,y,z):
    # points a, b and, c
    b1, b2, b3, b4, b5 = (0,0,0), (x,0,0), (x,y,0), (0,y,0), (0,0,0)
    B = np.array([b1, b2, b3, b4, b5])
    return B

# points of C rectangle of a hexahedron
def sixSideC(x,y,z):
    # points a, b and, c
    c1, c2, c3, c4, c5 = (0,0,z), (x,0,z), (x,0,0), (0,0,0), (0,0,z)
    C = np.array([c1, c2, c3, c4, c5])
    return C

# points of D rectangle of a hexahedron
def sixSideD(x,y,z):
    # points a, b and, c
    d1, d2, d3, d4, d5 = (x,y,z), (0,y,z), (0,y,0), (x,y,0), (x,y,z)
    D = np.array([d1, d2, d3, d4, d5])
    return D

# plotting a hexahedron with A,B,C,D
def plotting(A,B,C,D, c,a,m):
    ax.plot(A[:,0], A[:,1], A[:,2], color=c, alpha=a, marker=m)
    ax.plot(B[:,0], B[:,1], B[:,2], color=c, alpha=a, marker=m)
    ax.plot(C[:,0], C[:,1], C[:,2], color=c, alpha=a, marker=m)
    ax.plot(D[:,0], D[:,1], D[:,2], color=c, alpha=a, marker=m)

# 육면체의 선형변환들 클래스화
class Hexahedron():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
        self.A = sixSideA(x, y, z)
        self.B = sixSideB(x, y, z)
        self.C = sixSideC(x, y, z)
        self.D = sixSideD(x, y, z)
        
        self.newA = ArrayNew(self.A)
        self.newB = ArrayNew(self.B)
        self.newC = ArrayNew(self.C)
        self.newD = ArrayNew(self.D)
        
    # 선형변환들을 초기화시키고 원형으로 돌림
    def clear(self):
        self.newA = ArrayNew(self.A)
        self.newB = ArrayNew(self.B)
        self.newC = ArrayNew(self.C)
        self.newD = ArrayNew(self.D)
        
    def transpose(self, dx, dy, dz):
        self.newA = Ttr(dx, dy, dz, self.newA)
        self.newB = Ttr(dx, dy, dz, self.newB)
        self.newC = Ttr(dx, dy, dz, self.newC)
        self.newD = Ttr(dx, dy, dz, self.newD)
    
    def rotate_x(self, theta):
        self.newA = Trx(theta, self.newA)
        self.newB = Trx(theta, self.newB)
        self.newC = Trx(theta, self.newC)
        self.newD = Trx(theta, self.newD)
    
    def rotate_y(self, theta):
        self.newA = Try(theta, self.newA)
        self.newB = Try(theta, self.newB)
        self.newC = Try(theta, self.newC)
        self.newD = Try(theta, self.newD)
    
    def rotate_z(self, theta):
        self.newA = Trz(theta, self.newA)
        self.newB = Trz(theta, self.newB)
        self.newC = Trz(theta, self.newC)
        self.newD = Trz(theta, self.newD)
    
    def scale(self, sx, sy, sz):
        self.newA = Tscale(sx, sy, sz, self.newA)
        self.newB = Tscale(sx, sy, sz, self.newB)
        self.newC = Tscale(sx, sy, sz, self.newC)
        self.newD = Tscale(sx, sy, sz, self.newD)
    
    def shear(self, b, c, d, f, g, i):
        self.newA = Tsh(b, c, d, f, g, i, self.newA)
        self.newB = Tsh(b, c, d, f, g, i, self.newB)
        self.newC = Tsh(b, c, d, f, g, i, self.newC)
        self.newD = Tsh(b, c, d, f, g, i, self.newD)
        
    def draw_cube(self, c, a, m):
        plotting(self.newA, self.newB, self.newC, self.newD, c, a, m)
  
        
slime = Hexahedron(7, 7, 7)
slime.transpose(7, -10, -13)
slime.draw_cube("g", 0.6, "")


plt.show()