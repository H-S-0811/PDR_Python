import numpy as np
import matplotlib.pyplot as plt
import csv

# CSVファイルからデータを読み取る
csv_file = "output.csv"
data = []

with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        # 各行を数値データに変換してリストに追加
        data.append([float(value) for value in row])

# posi[0] を x, posi[1] を y にする
data = np.array(data)  # データを numpy 配列に変換
x = data[0]  # posi[0] (x座標)
y = data[1]  # posi[1] (y座標)

# グラフを描画
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Trajectory')

# グラフの装飾
plt.title("Trajectory of posi[0] (x) and posi[1] (y)")
plt.xlabel("X-axis (posi[0])")
plt.ylabel("Y-axis (posi[1])")
plt.grid(True)
plt.legend()
plt.tight_layout()

# グラフを表示
plt.show()