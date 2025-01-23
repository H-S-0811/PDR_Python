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


def update_Pk(Fk: np.ndarray, Pk_1: np.ndarray, Qk: np.ndarray) -> np.ndarray:
    Fk = Fk.copy()
    return (Fk @ Pk_1 @ Fk.T) + Qk


def update_Kk(H: np.ndarray, Pk: np.ndarray, R: np.ndarray) -> np.ndarray:
    k = np.linalg.inv((H @ Pk @ H.T) + R)
    return Pk @ H.T @ k


def update_Pk_in_ZUPT(Kk: np.ndarray, H: np.ndarray, Pk: np.ndarray) -> np.ndarray:
    return (np.identity(9, dtype=float) - (Kk @ H)) @ Pk


def update_C(Ck_1: np.ndarray, Ck_2: np.ndarray, Ck: np.ndarray):
    return Ck_1 @ np.linalg.inv(Ck_2) @ Ck


def func1(
    H: np.ndarray,
    Pk: np.ndarray,
    R: np.ndarray,
    vk: np.ndarray,
    pk: np.ndarray,
    Ck: np.ndarray,
    delta_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Kk = update_Kk(H, Pk, R)

    a = Kk @ vk
    Pk = update_Pk_in_ZUPT(Kk, H, Pk)
    # ここまで正解

    Omegaep_k = Omega_k(a[0:3])

    Ck_1_L = 2 * np.identity(3, dtype=np.float64) + Omegaep_k * delta_t
    Ck_2_L = 2 * np.identity(3, dtype=np.float64) - Omegaep_k * delta_t
    Ck = update_C(Ck_1_L, Ck_2_L, Ck)
    pk = pk - a[3:6]
    vk = vk - a[6:]
    return vk, pk, Pk, Ck


def func2(
    row: pd.Series,
    time_k: float,
    Ck_1: np.ndarray,
    vk_1: np.ndarray,
    gra: np.ndarray,
    Acc_N_1: np.ndarray,
    pk_1: np.ndarray,
    Pk_1: np.ndarray,
    Q: list[float],
    H: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    shape = (9, 1)
    A = np.zeros(shape)
    delta_t = row["time"] - time_k
    time_k = float(row["time"])
    Omegak = Omega(row)
    a_S = np.array([(row["acc_x"]), row["acc_y"], row["acc_z"]], dtype=np.float64)
    g_k = np.array(
        [row["gyr_x_calibrated"], row["gyr_y_calibrated"], row["gyr_z_calibrated"]],
        dtype=np.float64,
    )
    I = np.identity(3, dtype=np.float64)
    C_1 = 2 * I + (Omegak * delta_t)
    C_2 = 2 * I - (Omegak * delta_t)
    Ck = (Ck_1 @ C_1) @ np.linalg.inv(C_2)
    Acc_N = np.inner(0.5 * (Ck + Ck_1), a_S)
    Acc_N = Acc_N.reshape(3, 1)
    vk = vk_1 + ((Acc_N - gra) + (Acc_N_1 - gra)) * delta_t / 2
    pk = pk_1 + (vk + vk_1) * delta_t / 2

    Acc_N_1 = np.copy(Acc_N)
    Sk = S(Acc_N)
    Fk = F(Sk, delta_t)
    Qk = np.linalg.matrix_power(np.diag(Q) * (delta_t), 2)

    # Pk = np.matmul(Fk,Pk_1,Fk.T) + Qk
    Pk = update_Pk(Fk, Pk_1, Qk)

    if LA.norm(g_k, ord=2) < 0.6:
        vk, pk, Pk, Ck = func1(H=H, Pk=Pk, R=R, vk=vk, pk=pk, Ck=Ck, delta_t=delta_t)
    vk_1 = np.copy(vk)
    pk_1 = np.copy(pk)
    Pk_1 = np.copy(Pk)
    Ck_1 = np.copy(Ck)
    Acc_N_1 = np.copy(Acc_N)
    return vk_1, pk_1, Pk_1, Ck_1, pk, Acc_N_1, time_k
