import numpy as np
import math
from numpy import linalg as LA
import pandas as pd
import cul_func as Cul
import plotly.graph_objects as go

np.set_printoptions(suppress=True, precision=10)

csv_data = []
delta_t = 0
j = 0
posi = []
vel = []
ans = []
posi_E =[]
vel_E = []


# data = pd.read_csv("data/1.csv",sep=";")
data = pd.read_csv("data/iOS_data.csv",sep=";")
# data = data.rename(columns={
#     "acc_x": "acc_z",
#     "acc_y": "acc_x",
#     "acc_z": "acc_y",
#     "gyr_x": "gyr_z",
#     "gyr_y": "gyr_x",
#     "gyr_z": "gyr_y",

# })
data_Test = data.copy()
first = data.loc[0,:]
Omegak_1 = Cul.Omega(first)

# numpyの値表示設定
np.set_printoptions(formatter={'float': '{:.5e}'.format})

roll = math.atan(float(first.at["acc_y"])/float(first.at["acc_z"]))
pitch = -math.asin(float(first.at["acc_x"])/9.8)
C_x = [math.cos(pitch),math.sin(roll)*math.sin(pitch),math.cos(roll)*math.sin(pitch)]
C_y = [0,math.cos(roll),-math.sin(roll)]
C_z = [-math.sin(pitch),math.sin(roll)*math.cos(pitch),math.cos(roll)*math.cos(pitch)]
C_list = [C_x,C_y,C_z]
C = np.array(C_list,dtype=np.float64)
time_k =  np.copy(first["time"])
Ck_1 = C
A = np.array
a_S = np.array([(first["acc_x"]), first["acc_y"], first["acc_z"]],dtype=np.float64)
Acc_N_1 = np.inner(C,a_S)
Acc_N_1 = Acc_N_1.reshape(3,1)
vk_1 = np.array([[0],[0],[0]],dtype=np.float64)
pk_1 = np.array([[0],[0],[0]],dtype=np.float64)
Pk_1 = np.zeros((9,9),dtype=np.float64)
gra = np.array([[0],[0],[9.8]])
Q = [0.01,0.01,0.01,0,0,0,0.01,0.01,0.01]
r = np.array([0.01**2, 0.01**2, 0.01**2], dtype=np.float64)
R = np.diag(r)
H = np.hstack([np.hstack([np.zeros((3,3)),np.zeros((3,3))]),np.identity(3,dtype=np.float64)])

for index, row in data[1:].iterrows():
    A = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    delta_t = row["time"] - time_k
    time_k = np.copy(row["time"])
    # row["gyr_x"] = round(row["gyr_x"] - (-0.0156), 6)
    # row["gyr_y"] = round(row["gyr_y"] - (-0.0101), 6)
    # row["gyr_z"] = round(row["gyr_z"] - (-0.0020), 6)
    Omegak = Cul.Omega(row)
    a_S = np.array([row["acc_x"], row["acc_y"], row["acc_z"]],dtype=np.float64)
    g_k = np.array([row["gyr_x"],row["gyr_y"],row["gyr_z"]],dtype=np.float64)
    I = np.identity(3,dtype=np.float64)
    C_1 = 2 * I + (Omegak * delta_t)
    C_2 = 2 * I - (Omegak * delta_t)
    Ck = (Ck_1 @ C_1) @ np.linalg.inv(C_2)
    Acc_N = np.inner(0.5 * (Ck+Ck_1),a_S)
    Acc_N = Acc_N.reshape(3,1)
    # vk = vk_1 + ((Acc_N - gra) + (Acc_N_1 - gra))* delta_t / 2
    vk = vk_1 + ((Acc_N) + (Acc_N_1))* delta_t / 2
    pk = pk_1 + (vk+vk_1)* delta_t / 2

    Acc_N_1 = np.copy(Acc_N)
    Sk = Cul.S(Acc_N)
    Fk = Cul.F(Sk,delta_t)
    Qk = np.linalg.matrix_power(np.diag(Q)*(delta_t),2)

    Pk = Cul.Pk(Fk,Pk_1,Qk)

    if LA.norm(g_k,ord=2) < 0.6:
        Kk = Cul.Kk(H,Pk,R)

        a = Kk @ vk
        A = np.copy(a)
        Pk = Cul.update_Pk(Kk,H,Pk)
        # # ここまで正解
        Omegaep_k = Cul.Omega_k(a[0:3])

        Ck_1_L = (2 * np.identity(3,dtype=np.float64)) + Omegaep_k
        Ck_2_L = (2 * np.identity(3,dtype=np.float64)) - Omegaep_k
        Ck = Cul.update_C(Ck_1_L,Ck_2_L,Ck)
        pk = pk - a[3:6]
        vk = vk - a[6:]

    vk_1 = np.copy(vk)
    pk_1 = np.copy(pk)
    Pk_1 = np.copy(Pk)
    Ck_1 = np.copy(Ck)
    posi.append(pk)

#3D
for i in range(len(posi)):
    z= [posi[i][0][0],posi[i][1][0],posi[i][2][0]]
    ans.append(z)
ANS = pd.DataFrame(ans,columns=['X', 'Y','Z'])
fig = go.Figure(data=[
    go.Scatter3d(x=ANS["X"], y=ANS["Y"],z=ANS["Z"], name="acc_x", marker=dict(size=2)),
])

# 2D
# ans.append([0,0])
# for i in range(len(posi)):
#     g_time = data.loc[i,:]
#     time = g_time["time"]
#     XY= [time,posi[i][0][0],posi[i][1][0]]
#     ans.append(XY)
# POSI = pd.DataFrame(ans[0:],columns=["ind",'X', 'Y'])
# fig = go.Figure(data=[
#     go.Scatter(
#         mode='markers',
#         x=POSI["X"], 
#         y=POSI["Y"], 
#         name="MYPOSI",
#         text=POSI['ind'])
# ])
fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
fig.update_scenes(aspectmode='data')
fig.show()