#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:23:30 2023

@author: sairachawla
"""
import requests
import pandas as pd
url ='https://www.populationu.com/gen/us-states-by-population'
html = requests.get(url).content
df_lst = pd.read_html(html)
df = df_lst[0][['US States', 'Population 2022']]
df.to_csv('statepop2022.csv', sep='\t', encoding='utf-8')
