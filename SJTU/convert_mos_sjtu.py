import pandas as pd

df = pd.read_csv('SJTU/mos.csv')

all_mos = []
all_names = []

i = 0
for column in df.columns:
    all_mos.extend(df[column].to_list())
    all_names.extend([column] * len(df[column]))

new_df = pd.DataFrame({'name': all_names, 'mos': all_mos})

new_df['index'] = range(1,len(new_df)+1)
new_df['index'] = (new_df['index']-1)%42
new_df['name'] = new_df['name'] + '_'+ (new_df['index']).astype(str) + '.ply'
new_df = new_df.drop('index', axis=1)
new_df['dataset']= 'SJTU'

sorted_df = new_df.iloc[1:].sort_values(by='name', key=lambda x: x.str.lower().str[0:3])

sorted_df.to_csv('SJTU/SJTU_MOS.csv', index=False)