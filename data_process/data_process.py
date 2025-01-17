import numpy as np
import pandas as pd

def process(data,csv):
    start_time = pd.Timestamp('2015-12-22 16:30:00')
    time_index = pd.date_range(start=start_time, periods=data.shape[0], freq='S')

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 将时间编码添加为第一列
    df.insert(0, 'Time', time_index)
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)

    # 重命名最后一列为 'OT'
    df.rename(columns={df.columns[-1]: 'OT'}, inplace=True)

    df.to_csv(csv, index=False)

data = np.load('/home/wwr/zsn/dataset/MSL/MSL_train.npy')
data.astype("float")
csv = "/home/wwr/zsn/processed_data/MSL/MSL_train.csv"
process(data,csv)

data = np.load('/home/wwr/zsn/dataset/MSL/MSL_test.npy')
data.astype("float")
csv = "/home/wwr/zsn/processed_data/MSL/MSL_test.csv"
process(data,csv)