import pdb

import pandas as pd
import numpy as np
import json

with open("regions.json") as f:
    reg_name = json.load(f)

df = pd.read_csv('dpc-covid19-ita-regioni.csv')

regions = [i+1 for i in range(21)]  # list of regions to consider
predictors = ['newinfections', 'hospitalized', 'recovered', 'deceased']  # list of predictors to consider

df2 = df[['codice_regione', 'nuovi_positivi', 'totale_ospedalizzati', 'dimessi_guariti', 'deceduti']]
df2.columns = ['region', 'newinfections', 'hospitalized', 'recovered', 'deceased']
df2.at[22, 'region'] = 4

for reg in regions:

    region_rows = [i for i in range(len(df2['region'])) if df2['region'][i] == reg]
    region_df = df2.iloc[region_rows, 1:]
    region_name = reg_name[str(reg)]
    region_df.to_csv('./data_regions/' + region_name + '.csv')
