import pandas as pd
import csv

#read input file
df = pd.read_csv("WPC_NSS.csv")

sorted_df = df.sort_values(by="name", ascending=True)  

#set output fule
sorted_df.to_csv("WPC/WPC_NSS.csv", index=False)

print(sorted_df)