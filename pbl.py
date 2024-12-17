import csv
import numpy as np
import math
from numpy import linalg as LA

csv_data = []
delta_t: float = 1/120
j = 0
posi = []

def Cul_Omega(data: list) -> np.ndarray:
    x_data = [0,-1*float(data[6]),float(data[5])]
    y_data = [float(data[6]),0,-1*float(data[4])]
    z_data = [-1*float(data[5]),float(data[4]),0]
    M_list = [x_data,y_data,z_data]
    return np.array(M_list)

def Cul_Omega_k(data: np.ndarray) -> np.ndarray:
    x_data = [0,1 * float(data[2]), -float(data[1])]
    y_data = [-float(data[2]),0,1* float(data[0])]
    z_data = [1 *float(data[1]),-float(data[0]),0]
    M_list = [x_data,y_data,z_data]
    return np.array(M_list)

def Cul_S(data: np.ndarray) -> np.ndarray:
    x_data = [0, -1 * float(data[2, 0]), float(data[1, 0])]
    y_data = [float(data[2, 0]), 0, -1 * float(data[0, 0])]
    z_data = [-1 * float(data[1, 0]), float(data[0, 0]), 0]
    M_list = [x_data,y_data,z_data]
    return np.array(M_list)

def Cul_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != (3,3):
        raise Exception(a.shape)
    if not (b.shape == (3,3) or b.shape == (3,1)):
        raise Exception(b.shape)
    return a @ b

def Cul_acc_n(Ck: np.ndarray,Ck_1: np.ndarray,data: list) -> np.ndarray: 
    a_S = np.array(
        [float(data[1]), float(data[2]), float(data[3])]
    )
    a_S = a_S.reshape(3,1)
    
    if a_S.shape != (3,1):
        raise Exception(a_S.shape)

    return Cul_matrix((Ck+Ck_1),a_S)*1/2

def Cul_F(Sk: np.ndarray) -> np.ndarray:
    a = np.vstack([np.vstack([np.identity(3,dtype=float),np.zeros((3,3))]),-1 * Sk*delta_t])
    b = np.vstack([np.vstack([np.zeros((3,3)),np.identity(3,dtype=float)]),np.zeros((3,3))])
    c = np.vstack([np.vstack([np.zeros((3,3)),np.identity(3,dtype=float) * delta_t]),np.identity(3,dtype=float)])
    return np.concatenate([a,b,c], axis=1)


with open("data/1.csv") as f:
    reader = csv.reader(f,delimiter =';')
    i = 0
    for row in reader:
        csv_data.append(row)
        i = i+1

Omegak_1 = Cul_Omega(csv_data[1])
roll = math.atan(float(csv_data[1][2])/float(csv_data[1][3]))
pitch = -math.asin(float(csv_data[1][1])/9.8)
C_x = [math.cos(pitch),math.sin(roll)*math.sin(pitch),math.cos(roll)*math.sin(pitch)]
C_y = [0,math.cos(roll),-math.sin(roll)]
C_z = [-math.sin(pitch),math.sin(roll)*math.cos(pitch),math.cos(roll)*math.cos(pitch)]
C_list = [C_x,C_y,C_z]
C = np.array(C_list)

# Ck_1 = Cul_matrix(2 * np.identity(3,dtype=float) + Omegak_1 * delta_t , np.linalg.inv(2 * np.identity(3,dtype=float) - Omegak_1*delta_t))
Ck_1 = C
# Acc_N_1 = Cul_acc_n(C,Ck_1,csv_data[1])
a_S = np.array([float(csv_data[1][1]), float(csv_data[1][2]), float(csv_data[1][3])])
Acc_N_1 = C @ a_S
vk_1 = np.array([[0],[0],[0]])
pk_1 = np.array([[0],[0],[0]])
Pk_1 = np.zeros((9,9))
gra = np.array([[0],[0],[9.8]])
Q = [0.01,0.01,0.01,0,0,0,0.01,0.01,0.01]
Qk = np.linalg.matrix_power(np.diag(Q)*delta_t,2)
r = [0.01**2,0.01**2,0.01**2]
R = np.diag(r)
H = np.hstack([np.hstack([np.zeros((3,3)),np.zeros((3,3))]),np.identity(3,dtype=float)])

for i in range(2,len(csv_data)):
# for i in range(2,5):
    delta_t = float(csv_data[i][0]) - float(csv_data[i-1][0])
    csv_data[i][4] = str(float(csv_data[i][4]) - (-0.0156))
    csv_data[i][5] = str(float(csv_data[i][5]) - (-0.0101))
    csv_data[i][6] = str(float(csv_data[i][6]) - (-0.0020))
    Omegak = Cul_Omega(csv_data[i])
    g_k = np.array([csv_data[i][4],csv_data[i][5],csv_data[i][6]])
    Ck = Ck_1 @ (2 * np.identity(3,dtype=float) + (Omegak * delta_t)) @ np.linalg.inv(2 * np.identity(3,dtype=float) - (Omegak*delta_t))


    Acc_N = Cul_acc_n(Ck,Ck_1,csv_data[i])
    vk = vk_1 + (Acc_N + Acc_N_1 - 2*gra)* delta_t / 2
    pk = pk_1 + (vk+vk_1)* delta_t / 2
    # print(f"{pk[0][0]},{pk[1][0]},{pk[2][0]}")
    
    Acc_N_1 = np.copy(Acc_N)
    vk_1 = np.copy(vk)
    pk_1 = np.copy(pk)
    Sk = Cul_S(Acc_N)
    Fk = Cul_F(Sk)
    Pk = np.matmul(Fk,Pk_1,Fk.T) + Qk
    # if LA.norm(g_k) < 0.6:
    #     k = np.linalg.inv(H @ Pk @ H.T + R)
    #     Kk = Pk @ H.T @ k
    #     a = Kk @ vk
    #     Pk = (np.identity(9,dtype=float) - Kk @ H) @ Pk
    #     Omegaep_k = Cul_Omega_k(a)
    #     Ck = Cul_matrix(2 * np.identity(3,dtype=float) + Omegaep_k * delta_t , 2 * np.identity(3,dtype=float) - Omegaep_k*delta_t)

    #     pk = pk - a[3:6]
    #     vk = vk - a[6:]
    #     vk_1 = vk
    #     pk_1 = pk
    
    Pk_1 = np.copy(Pk)
    Ck_1 = np.copy(Ck)
    posi.append(pk)
