import pandas as pd


a = pd.read_csv('/data0/EEV/code/baseline/en_tv0.csv')
b = pd.read_csv('/data0/EEV/code/baseline/en_tv1.csv')
c = pd.read_csv('/data0/EEV/code/baseline/en_tv2.csv')
d = pd.read_csv('/data0/EEV/code/baseline/en_tv3.csv')
e = pd.read_csv('/data0/EEV/code/baseline/en_tv4.csv')

f = pd.read_csv('/data0/EEV/code/baseline/en_tv5.csv')
g = pd.read_csv('/data0/EEV/code/baseline/en_tv6.csv')
h = pd.read_csv('/data0/EEV/code/baseline/en_tv7.csv')
i = pd.read_csv('/data0/EEV/code/baseline/en_tv8.csv')
j = pd.read_csv('/data0/EEV/code/baseline/en_tv9.csv')

ensemble = pd.read_csv('/data0/EEV/code/tools/ensemble.csv')

outputs = [a, b, c, d, e, f, g, h, i, j]
# config = [
#     [0, 1, 0, 0, 0, 0], # 0 amusement...
#     [0, 0, 1, 1, 1, 0],
#     [1, 0, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1, 1],
#     [1, 2, 1, 0, 0, 0],
#     [1, 0, 0, 0, 0, 2], # 5
#     [0, 1, 1, 1, 0, 0],
#     [1, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 1],
#     [1, 1, 0, 0, 1, 0], # 10
#     [0, 0, 1, 0, 0, 1],
#     [0, 0, 2, 0, 1, 0],
#     [0, 0, 0, 0, 2, 1],
#     [0, 0, 0, 1, 1, 1]
# ]

config = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 0 amusement...
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 5
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 10
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

ensemble.iloc[:, 2:] = 0.0
for idx, e in enumerate(config):
    scale = sum(e)
    print('Emotion:', idx, 'Scale:', scale)
    for d_idx, w in enumerate(e):
        ensemble.iloc[:, 2 + idx] += (w/scale) * outputs[d_idx].iloc[:, 2 + idx]

with open('ensemble.csv', 'w') as f:
    ensemble.to_csv(f, index=False)