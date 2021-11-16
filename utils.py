import pandas as pd
from pandas.core.frame import DataFrame
import itertools
import numpy as np

def readFile(path:str)-> pd.DataFrame:
    df = pd.read_csv(path)
    df['rec_id'] = range(1, len(df) +1)
    return df

def separate_data(df:pd.DataFrame, outcome:str):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    out = df[outcome]

    attributes = {}
    for column in df:
        if column == 'rec_id':
            continue
        elif column == outcome:
            tmp = df[[column,'rec_id']]
            tmp['rec_id'] = range(1, len(df)+1 )
            tmp['leaf'] = ['N1' for _ in range(len(df))]
            attributes[column] = tmp
        else:
            tmp = df[[column,'rec_id']]
            tmp['rec_id'] = range(1, len(df)+1 )
            tmp[outcome] = df[outcome]
            if column != outcome:
                tmp = tmp.sort_values(by=[column])
            attributes[column] = tmp

    return attributes

def gini_attr(attr:DataFrame, attr_name:str, outcome:str):
    if list(attr.columns.values)[0] == outcome:
        return
    gini_T = gini(attr[outcome])
    N = len(attr)
    gini_values = {}
    for i in range(1,N):
        l = attr[outcome][:i]
        l_frame = attr[:1]
        n_l = len(l)
        r = attr[outcome][i:]
        r_frame = attr[i:]
        n_r = len(r)
        l_gini = gini(l)
        r_gini = gini(r)
        gini_Ta = gini_mean(l_gini,n_l,r_gini,n_r,N)
        gini_values[i] =gini_T - gini_Ta
    if not gini_values:
        return 
    sep_idx = max(gini_values, key=gini_values.get)
    tmp = attr.reset_index(drop=True)
    print('N ->  <=', tmp[attr_name][sep_idx-1],'\n')
    gini_attr(l_frame, attr_name, outcome)
    gini_attr(r_frame, attr_name, outcome)
    return sep_idx

def gini(df):
    values = df.value_counts(dropna=False)
    n = len(df) 
    sq_sum = 0
    for i in values.iteritems():
        sq_sum += (i[1] / n)**2
    return  1 - sq_sum

def gini_mean(l,n_l, r,n_r,N):
    t1 = (n_l/N)*l 
    t2 = (n_r/N)*r
    return t1+t2

