import pandas as pd

df = pd.read_csv('WPC2/mos.csv')

df = df.drop('content', axis=1)
df = df.drop('geo_QP', axis=1)
df = df.drop('col_QP', axis=1)
df['dataset'] = "WPC2"
df.to_csv("WPC2/WPC2_MOS.csv", index=False)