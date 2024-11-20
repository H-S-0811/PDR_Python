import csv
import numpy as np

csv_data = []
delta_t: float = 1/120


def Cul_Omega(data: list) -> np.ndarray:
    x_data = [0,-1 * float(data[6]),float(data[5])]
    y_data = [float(data[6]),0,-1* float(data[4])]
    z_data = [-1 *float(data[5]),float(data[4]),0]
    M_list = [x_data,y_data,z_data]
    return np.array(M_list)

def Cul_S(data: np.ndarray) -> np.ndarray:
    x_data = [0,-1 * float(data[2]),float(data[1])]
    y_data = [float(data[2]),0,-1* float(data[0])]
    z_data = [-1 *float(data[1]),float(data[0]),0]
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
    c = np.vstack([np.vstack([np.zeros((3,3)),delta_t * np.identity(3,dtype=float)]),np.identity(3,dtype=float)])
    return np.concatenate([a,b,c], axis=1)

    

with open("data/1.csv") as f:
    reader = csv.reader(f,delimiter =';')
    i = 0
    for row in reader:
        csv_data.append(row)
        i = i+1

Omegak_1 = Cul_Omega(csv_data[1])
Ck_1 = Cul_matrix(2 * np.identity(3,dtype=float) + Omegak_1 * delta_t , np.linalg.inv(2 * np.identity(3,dtype=float) - Omegak_1*delta_t))
Acc_N_1 = Cul_acc_n(Ck_1,Ck_1,csv_data[1])
vk_1 = np.array([[0],[0],[0]])
pk_1 = np.array([[0],[0],[0]])
P = np.zeros((9,9))
gra = np.array([[0],[0],[9.8]])

for i in range(2,120):
    Omegak = Cul_Omega(csv_data[i])
    Ck = Cul_matrix(2 * np.identity(3,dtype=float) + Omegak * delta_t , 2 * np.identity(3,dtype=float) - Omegak*delta_t)
    Acc_N = Cul_acc_n(Ck,Ck_1,csv_data[i])
    vk = vk_1 + (Acc_N + Acc_N_1 - 2*gra)* delta_t / 2
    pk = pk_1 + (vk+vk_1)* delta_t / 2
    print(f"{pk[0][0]},{pk[1][0]},{pk[2][0]}")
    Ck_1 = Ck
    Acc_N_1 = Acc_N
    vk_1 = vk
    pk_1 = pk
    Sk = Cul_S(Acc_N)
    Fk = Cul_F(Sk)
   

