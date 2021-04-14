import pandas as pd

exp = ['8pZ_1NW7WJo',
'DB2io5Amlxo',
'fjJYTW2n6rk',
'GufMoL_MuNE',
'_jFBeLJsxJI',
'L9cdaj74kLo',
'l-ka23gU4NA',
'o0ooW14pIa4',
'QMW3GuohzzE',
'R9kJlLungmo',
'rbTIMt0VcLw',
'ScvvOWtb04Q',
'Uee0Tv1rTz8',
'WKXrnB7alT8',
'ZNOnRu7GeBc']
exp = set(exp)

c1 = pd.read_csv('/data0/EEV/code/baseline/test_output.csv')
c2 = pd.read_csv('/data0/EEV/eev-csv/test.csv')

v1 = set(c1['Video ID'].to_list())
v2 = set(c2['Video ID'].to_list())

print('e - 1, pass?', len(exp - v1) <= len(exp))
print('2 - 1', v2 - v1 - exp)
print(list(v2 - v1))