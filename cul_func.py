import numpy as np
import math
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt

def Omega(data: pd.Series|pd.DataFrame) -> np.ndarray:
    x_data = [0,-1*data["gyr_z"],data["gyr_y"]]
    y_data = [data["gyr_z"],0,-1*data["gyr_x"]]
    z_data = [-1*data["gyr_y"],data["gyr_x"],0]
    M_list = [x_data,y_data,z_data]
    return np.array(M_list)

def Omega_k(data: np.ndarray) -> np.ndarray:
    x_data = [0,1 * float(data[2]), -float(data[1])]
    y_data = [-float(data[2]),0,1* float(data[0])]
    z_data = [1 *float(data[1]),-float(data[0]),0]
    M_list = [x_data,y_data,z_data]
    return np.array(M_list)

def S(data: np.ndarray) -> np.ndarray:
    x_data = [0, -1 * float(data[2, 0]), float(data[1, 0])]
    y_data = [float(data[2, 0]), 0, -1 * float(data[0, 0])]
    z_data = [-1 * float(data[1, 0]), float(data[0, 0]), 0]
    M_list = [x_data,y_data,z_data]
    return np.array(M_list)

def Matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != (3,3):
        raise Exception(a.shape)
    if not (b.shape == (3,3) or b.shape == (3,1)):
        raise Exception(b.shape)
    return a @ b

def Acc_n(Ck: np.ndarray,Ck_1: np.ndarray,data:pd.Series|pd.DataFrame) -> np.ndarray: 
    a_S = np.array(
        [float(data.at["acc_x"]), float(data.at["acc_y"]), float(data.at["acc_z"])]
    )
    a_S = a_S.reshape(3,1)
    
    if a_S.shape != (3,1):
        raise Exception(a_S.shape)

    return Matrix((Ck+Ck_1),a_S)*1/2

def F(Sk: np.ndarray, delta_t: float) -> np.ndarray:
    if Sk.shape != (3,3):
        raise Exception(Sk.shape)
    a = np.vstack([np.vstack([np.identity(3,dtype=float),np.zeros((3,3))]),-1 * delta_t * Sk])
    b = np.vstack([np.vstack([np.zeros((3,3)),np.identity(3,dtype=float)]),np.zeros((3,3))])
    c = np.vstack([np.vstack([np.zeros((3,3)),np.identity(3,dtype=float) * delta_t]),np.identity(3,dtype=float)])
    return np.concatenate([a,b,c], axis=1)

def Pk(Fk: np.ndarray,Pk_1: np.ndarray,Qk: np.ndarray) -> np.ndarray:
    Fk = Fk.copy()
    return (Fk @ Pk_1 @ Fk.T) + Qk 

def Kk(H: np.ndarray, Pk: np.ndarray, R: np.ndarray) -> np.ndarray:
    k_1 = H @ Pk
    k_2 = k_1 @ H.T
    k_3 = k_2 + R
    k_4 = np.linalg.inv(k_3)
    k_5 = Pk @ H.T
    return k_5 @ k_4 
    # k_1 = Pk @ H.T
    # k_2 = H @ Pk @ H.T) + R
    # K_k = np.linalg.solve(k_2.T, k_1.T).T
    # return K_k


def update_Pk(Kk: np.ndarray, H: np.ndarray, Pk: np.ndarray) -> np.ndarray:
    p_1 = Kk @ H
    p_2 = np.identity(9,dtype=float) - p_1
    p_3 = p_2 @ Pk
    return p_3
    # return (np.identity(9,dtype=float) - (Kk @ H)) @ Pk

def update_C(Ck_1 :np.ndarray, Ck_2 : np.ndarray, Ck : np.ndarray):
    c_1 = np.linalg.inv(Ck_2)
    c_2 = Ck_1 @ c_1
    c_3 = c_2 @ Ck
    # result =  Ck_1 @ np.linalg.inv(Ck_2) @ Ck
    return np.round(c_3, 15)