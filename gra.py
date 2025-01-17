import pandas as pd
import numpy as np
import plotly.graph_objects as go



data = pd.read_csv("data/1.csv",sep=';')
print(type(data["time"]))
fig = go.Figure(data=[
    go.Scatter(x=data["time"], y=data["acc_x"], name="acc_x"),
    go.Scatter(x=data["time"], y=data["acc_y"], name="acc_y"),
    go.Scatter(x=data["time"], y=data["acc_z"], name="acc_z"),
])
# fig.show()