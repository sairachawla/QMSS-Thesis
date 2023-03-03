#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:11:01 2023

@author: sairachawla
"""

## importing necessary libraries
import pandas as pd
import numpy as np
import requests

## importing first dataset that describes healthcare and abortion related features in 2018
df_2018 = pd.read_excel('plosone/plosonedata.xlsx')
## cleaning the data
columnNames = df_2018.iloc[0] 
df_2018 = df_2018[1:] 
df_2018.columns = columnNames
df_2018 = df_2018.drop([52, 53])
for i in df_2018.columns[1:]:
    df_2018[i] = pd.to_numeric(df_2018[i])
df_2018['% Opine that abortion should be illegal'] = 100*df_2018['% Opine that abortion should be illegal']
df_2018['% Population 18-24 years old'] = 100*df_2018['% Population 18-24 years old']

df_2018.to_csv('plosonedata2018.csv', encoding='utf-8')