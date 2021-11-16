from utils import *

df = readFile('./dataset/paper_some.csv')
attributes = separate_data(df, 'Class')
for attr in attributes:
    if attr != 'Class':
        tmp = gini_attr(attr=attributes[attr], attr_name=attr, outcome='Class')
pass