import numpy as np
import math
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt


def Omega(data: pd.Series | pd.DataFrame) -> np.ndarray:
    data = data.copy()
    x_data = [0, -1 * data["gyr_z"], data["gyr_y"]]
    y_data = [data["gyr_z"], 0, -1 * data["gyr_x"]]
    z_data = [-1 * data["gyr_y"], data["gyr_x"], 0]
    M_list = [x_data, y_data, z_data]
    return np.array(M_list)


def Omega_k(data: np.ndarray) -> np.ndarray:
    x_data = [0, 1 * float(data[2]), -float(data[1])]
    y_data = [-float(data[2]), 0, 1 * float(data[0])]
    z_data = [1 * float(data[1]), -float(data[0]), 0]
    M_list = [x_data, y_data, z_data]
    return np.array(M_list)


def S(data: np.ndarray) -> np.ndarray:
    x_data = [0, -1 * float(data[2, 0]), float(data[1, 0])]
    y_data = [float(data[2, 0]), 0, -1 * float(data[0, 0])]
    z_data = [-1 * float(data[1, 0]), float(data[0, 0]), 0]
    M_list = [x_data, y_data, z_data]
    return np.array(M_list)


def Matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != (3, 3):
        raise Exception(a.shape)
    if not (b.shape == (3, 3) or b.shape == (3, 1)):
        raise Exception(b.shape)
    return a @ b


def Acc_n(
    Ck: np.ndarray, Ck_1: np.ndarray, data: pd.Series | pd.DataFrame
) -> np.ndarray:
    data = data.copy()
    a_S = np.array(
        [float(data.at["acc_x"]), float(data.at["acc_y"]), float(data.at["acc_z"])]
    )
    a_S = a_S.reshape(3, 1)

    if a_S.shape != (3, 1):
        raise Exception(a_S.shape)

    return Matrix((Ck + Ck_1), a_S) * 1 / 2


def F(Sk: np.ndarray, delta_t: float) -> np.ndarray:
    if Sk.shape != (3, 3):
        raise Exception(Sk.shape)
    a = np.vstack(
        [np.vstack([np.identity(3, dtype=float), np.zeros((3, 3))]), -1 * delta_t * Sk]
    )
    b = np.vstack(
        [np.vstack([np.zeros((3, 3)), np.identity(3, dtype=float)]), np.zeros((3, 3))]
    )
    c = np.vstack(
        [
            np.vstack([np.zeros((3, 3)), np.identity(3, dtype=float) * delta_t]),
            np.identity(3, dtype=float),
        ]
    )
    return np.concatenate([a, b, c], axis=1)


def Pk(Fk: np.ndarray, Pk_1: np.ndarray, Qk: np.ndarray) -> np.ndarray:
    Fk = Fk.copy()
    return (Fk @ Pk_1 @ Fk.T) + Qk


def Kk(H: np.ndarray, Pk: np.ndarray, R: np.ndarray) -> np.ndarray:
    k = np.linalg.inv((H @ Pk @ H.T) + R)
    return Pk @ H.T @ k


def update_Pk(Kk: np.ndarray, H: np.ndarray, Pk: np.ndarray) -> np.ndarray:
    return (np.identity(9, dtype=float) - (Kk @ H)) @ Pk


def update_C(Ck_1: np.ndarray, Ck_2: np.ndarray, Ck: np.ndarray):
    return Ck_1 @ np.linalg.inv(Ck_2) @ Ck
