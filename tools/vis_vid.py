from tensorboardX import SummaryWriter
import pandas as pd

vid = '--fcq9ziOpE'
writer = SummaryWriter(log_idr='/data0/EEV/code/tools/log', 'vis' + vid)

content = pd.read_csv('/data0/EEV/eev-csv/train.csv')
vlist = content['Youtube ID'].to_list()
