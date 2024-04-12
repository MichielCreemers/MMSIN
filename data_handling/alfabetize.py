import pandas as pd
import csv

#read input file
df = pd.read_csv("WPC/WPC_MOS.csv")

sorted_df = df.sort_values(by="Ply_name", ascending=True)  

#set output fule
sorted_df.to_csv("WPC/WPC_MOS.csv", index=False)

print(sorted_df)