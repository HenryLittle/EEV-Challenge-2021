import pandas as pd

# 0.030
a = pd.read_csv('/data0/EEV/code/baseline/test_output_1.csv')
# merge-100
b = pd.read_csv('/data0/EEV/code/baseline/test_output.csv')


idxa = [1, 3, 4, 5, 6, 11, 12, 13]
idxb = [0, 2, 7, 8, 9, 10, 14]
# Awe, Confusion, Elation, Interest, Pain, Sadness, Anger
idxb = [2, 4, 9, 10, 11, 12, 1]
idxb = [x + 2 for x in idxb] # skip the vid column

for idx in idxb:
    print(idx)
    col = b.iloc[:, idx]
    a.iloc[:, idx] = col

with open('ensemble.csv', 'w') as f:
    a.to_csv(f, index=False)
    