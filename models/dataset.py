import torch
import time
import numpy as np
from torch.utils.data import Dataset


# 시간을 window_size 단위로 자르기
def make_data_idx(dates, window_size=1):
    input_idx = []
    for idx in range(window_size - 1, len(dates)):
        _in_period = dates[idx] - dates[idx - (window_size - 1)]

        # 인덱스 생성
        if _in_period == (window_size - 1):
            input_idx.append(list(range(idx - window_size + 1, idx + 1)))
    return input_idx


class TagDataset(Dataset):
    def __init__(self, input_size, df, mean_df=None, std_df=None, window_size=1):
        pass_col = ['created_dt', 'date', 'idx', 'daq_id', 'label']
        self.input_size = input_size
        self.window_size = window_size
        original_df = df.copy()

        # 정규화
        if mean_df is not None and std_df is not None:
            sensor_columns = [item for item in original_df.columns if item not in pass_col]
            original_df[sensor_columns] = (df[sensor_columns] - mean_df) / std_df

        # 입력 데이터셋을 window 단위로 시퀀스 인덱스 생성
        dates = original_df['created_dt'].to_list()
        self.input_ids = make_data_idx(dates, window_size=window_size)
        # 독립변수 설정(센서 데이터)
        self.selected_column = [item for item in original_df.columns if item not in pass_col][:input_size]
        # np.type 변경
        before_data = np.array(original_df[self.selected_column].values, dtype=np.float64)
        self.var_data = torch.tensor(before_data).float()
        self.df = original_df.iloc[np.array(self.input_ids)[:, -1]]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        temp_input_ids = self.input_ids[item]
        input_values = self.var_data[temp_input_ids]
        return input_values