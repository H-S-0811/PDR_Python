import pandas as pd

a = pd.read_csv("a.csv")
ans = pd.read_csv("ans.csv")

a.columns = ["X","Y","Z"]
ans.columns = ["X","Y","Z"]

diff = ans-a

data = pd.read_csv("data/1.csv",sep=";")
first = data.loc[0,:]

print(first)
# print(diff.iloc[1:50])
