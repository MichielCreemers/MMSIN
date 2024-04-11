##implementation of min_max_scaler
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('WPC/WPC_NSS.csv')

scaler = MinMaxScaler()

feature_columns = df.columns[1:]

normalized_data = scaler.fit_transform(df[feature_columns])

normalized_dataframe = pd.DataFrame(normalized_data, columns=feature_columns)

normalized_dataframe.insert(0,'name', df['name'])

print(normalized_dataframe)

normalized_dataframe.to_csv("WPC/WPC_NSS_nor")