import numpy as np
import math
from numpy import linalg as LA
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=10)

data_A = pd.read_csv("Accelerometer.csv",sep=",")
data_G = pd.read_csv("Gyroscope.csv",sep=",")

# data_A = data_A.drop(columns="time").rename(columns={
#     "seconds_elapsed": "time",
#     "x": "acc_y",
#     "y": "acc_z",
#     "z": "acc_x",
# })
data_A = data_A.drop(columns="time").rename(columns={
    "seconds_elapsed": "time",
    "x": "acc_x",
    "y": "acc_y",
    "z": "acc_z",
})


# data_G = data_G.drop(columns=["time"]).rename(columns={
#     "seconds_elapsed": "time",
#     "x": "gyr_y",
#     "y": "gyr_z",
#     "z": "gyr_x",
# })
data_G = data_G.drop(columns=["time"]).rename(columns={
    "seconds_elapsed": "time",
    "x": "gyr_x",
    "y": "gyr_y",
    "z": "gyr_z",
})
print(data_A)
print(data_G)
data_A = data_A.reindex(columns=['time', 'acc_x', 'acc_y','acc_z'])
data_G = data_G.reindex(columns=['time', 'gyr_x', 'gyr_y','gyr_z'])
# time を基準にデータを結合（内部結合）
merged_data = pd.merge(data_A, data_G, on="time", how="inner")
print(merged_data)

# plt.figure(figsize=(8, 5))
# plt.plot(data_A["time"], data_A["acc_x"], marker='o', linestyle='-', color='b', label='acc_x')
# plt.plot(data_A["time"], data_A["acc_z"], marker='o', linestyle='-', color='r', label='acc_z')
# plt.xlabel("Time (s)")
# plt.ylabel("Acceleration X")
# plt.title("Time vs Acceleration X")
# plt.legend()
# plt.grid()
# plt.show()
# # CSVファイルとして保存（セミコロン区切り）
merged_data.to_csv("/Users/hori/pbl/data/iOS_data.csv", index=False, sep=";")