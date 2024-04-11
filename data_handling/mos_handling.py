import csv
import pandas as pd

df = pd.read_csv("../WPC_MOS.csv")

df['MOS'] = df['MOS']/100

df.to_csv("../WPC_MOS.csv", index=False)