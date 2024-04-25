##implementation of min_max_scaler
import joblib
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('nss_non_scaled.csv')

scaler = MinMaxScaler()

feature_columns = df.columns[1:]

normalized_data = scaler.fit_transform(df[feature_columns])

scaler.fit(df[feature_columns])

min_values = scaler.min_
scale = scaler.scale_

scaler_params = np.array([min_values, scale])

np.save('scaler_params.npy', scaler_params)

normalized_dataframe = pd.DataFrame(normalized_data, columns=feature_columns)

normalized_dataframe.insert(0,'name', df['name'])

print(normalized_dataframe)

normalized_dataframe.to_csv("nss_scaled")


df = pd.read_csv('nss_non_scaled.csv')

scaler_params = np.load('scaler_params.npy')

# Create a new MinMaxScaler object and set its parameters for multiple features
scaler_loaded = MinMaxScaler()
scaler_loaded.min_ = scaler_params[0]
scaler_loaded.scale_ = scaler_params[1]

feature_columns = df.columns[1:]

normalized_data_2 = scaler_loaded.transform(df[feature_columns])

normalized_dataframe_2 = pd.DataFrame(normalized_data_2, columns=feature_columns)

normalized_dataframe_2.insert(0,'name', df['name'])

print(normalized_dataframe_2)

normalized_dataframe_2.to_csv("nss_scaled_joblib")