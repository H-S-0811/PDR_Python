import csv
import numpy as np
import math
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import cul_func as Cul
import sys
import plotly.graph_objects as go

np.set_printoptions(suppress=True, precision=10)

csv_data = []
delta_t = 0
j = 0
posi = []
vel = []
ans = []
posi_E = []
vel_E = []

data = pd.read_csv("data/1.csv", sep=";")
data.loc[:, "gyr_x_calibrated"] = data.loc[:, "gyr_x"] - (-0.0156)
data.loc[:, "gyr_y_calibrated"] = data.loc[:, "gyr_y"] - (-0.0101)
data.loc[:, "gyr_z_calibrated"] = data.loc[:, "gyr_z"] - (-0.0020)
data_Test = data.copy()
first = data.iloc[0]
Omegak_1 = Cul.Omega(first)

# numpyの値表示設定
np.set_printoptions(formatter={"float": "{:.5e}".format})

roll = math.atan(float(first.at["acc_y"]) / float(first.at["acc_z"]))
pitch = -math.asin(float(first.at["acc_x"]) / 9.8)
C_x = [
    math.cos(pitch),
    math.sin(roll) * math.sin(pitch),
    math.cos(roll) * math.sin(pitch),
]
C_y = [0, math.cos(roll), -math.sin(roll)]
C_z = [
    -math.sin(pitch),
    math.sin(roll) * math.cos(pitch),
    math.cos(roll) * math.cos(pitch),
]
C_list = [C_x, C_y, C_z]
C = np.array(C_list, dtype=np.float64)
time_k = float(first["time"])
Ck_1 = C
a_S = np.array([(first["acc_x"]), first["acc_y"], first["acc_z"]], dtype=np.float64)
Acc_N_1 = np.inner(C, a_S)
Acc_N_1 = Acc_N_1.reshape(3, 1)
vk_1 = np.array([[0], [0], [0]], dtype=np.float64)
pk_1 = np.array([[0], [0], [0]], dtype=np.float64)
Pk_1 = np.zeros((9, 9), dtype=np.float64)
gra = np.array([[0], [0], [9.8]])
Q = [0.01, 0.01, 0.01, 0, 0, 0, 0.01, 0.01, 0.01]
r = [0.01**2, 0.01**2, 0.01**2]
R = np.diag(r)
H = np.hstack(
    [np.hstack([np.zeros((3, 3)), np.zeros((3, 3))]), np.identity(3, dtype=np.float64)]
)

for index, row in data[1:].iterrows():
    vk_1, pk_1, Pk_1, Ck_1, pk, Acc_N_1, time_k = Cul.func2(
        row=row,
        time_k=time_k,
        Ck_1=Ck_1,
        vk_1=vk_1,
        gra=gra,
        Acc_N_1=Acc_N_1,
        pk_1=pk_1,
        Pk_1=Pk_1,
        Q=Q,
        H=H,
        R=R,
    )

    posi.append(pk)

# print(pd.testing.assert_frame_equal(data,data_Test))

# エラー値の差分を調査
# posi_E = np.array(posi_E)
# posi_E = posi_E.reshape(posi_E.shape[1], posi_E.shape[2])
# po_E = pd.DataFrame(posi_E)
# po_E.to_csv("po_E.csv")

# vel_E = np.array(vel_E)
# vel_E= vel_E.reshape(vel_E.shape[1], vel_E.shape[2])
# ve_E = pd.DataFrame(vel_E)
# ve_E.to_csv("ve_E.csv")

# 3D
# for i in range(len(posi)):
#     z= [posi[i][0][0],posi[i][1][0],posi[i][2][0]]
#     ans.append(z)
# ans = pd.read_csv("ans.csv")
# ans.columns = ["X","Y","Z"]
# ANS = pd.DataFrame(ans,columns=['X', 'Y','Z'])
# fig = go.Figure(data=[
#     go.Scatter3d(x=ANS["X"], y=ANS["Y"],z=ANS["Z"], name="acc_x", marker=dict(size=2)),
# ])

# 2D
for i in range(len(posi)):
    z = [posi[i][0][0], posi[i][1][0]]
    ans.append(z)
# ans = pd.read_csv("ans.csv")
# ans.columns = ["X","Y","Z"]
ANS = pd.DataFrame(ans[1:], columns=["X", "Y"])
fig = go.Figure(
    data=[
        go.Scatter(x=ANS["X"], y=ANS["Y"]),
    ]
)
fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)
fig.update_scenes(aspectmode="data")
fig.show()
