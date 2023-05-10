#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:17:33 2023

@author: sairachawla
"""

import requests
import pandas as pd
url ='https://www.faa.gov/air_traffic/publications/atpubs/cnt_html/appendix_a.html'
html = requests.get(url).content
df_lst = pd.read_html(html)
df = df_lst[0]
df1 = df[['STATE(TERRITORY)', 'STATE(TERRITORY).1']].rename(columns={'STATE(TERRITORY)':'State', 'STATE(TERRITORY).1':'Abbrev'})
df2 = df[['STATE(TERRITORY).2', 'STATE(TERRITORY).3']].rename(columns={'STATE(TERRITORY).2':'State', 'STATE(TERRITORY).3':'Abbrev'})
df3 = df[['STATE(TERRITORY).4', 'STATE(TERRITORY).5']].rename(columns={'STATE(TERRITORY).4':'State', 'STATE(TERRITORY).5':'Abbrev'})
df = pd.concat([df1, df2, df3])
df = df.dropna().reset_index()[['State', 'Abbrev']]
df.to_csv('stateabbrevs.csv', sep='\t', encoding='utf-8')