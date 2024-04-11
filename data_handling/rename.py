
import pandas as pd

df = pd.read_csv('WPC/WPC_NSS.csv')

#split, and keep only the right side of the last /
df['name'] = df['name'].str.rsplit('/', n=1).str[-1]

#write to csv
df.to_csv('WPC/WPC_NSS.csv', index=False)

     