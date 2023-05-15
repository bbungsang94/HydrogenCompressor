import copy
import os
import numpy as np
import pandas as pd


def get_features(data: pd.Series):
    np_data = data.to_numpy()
    statistic_dict = dict()
    # statistics
    statistic_dict['data_mean'] = np.mean(np_data)
    statistic_dict['data_median'] = np.median(np_data)
    statistic_dict['data_max'] = np.max(np_data)
    statistic_dict['data_min'] = np.min(np_data)
    statistic_dict['data_std'] = np.std(np_data)
    statistic_dict['data_var'] = np.var(np_data)
    statistic_dict['data_kurt'] = data.kurt()
    statistic_dict['data_peak'] = data.skew()
    statistic_dict['data_prod'] = data.prod()
    # combination
    statistic_dict['comb_sra'] = np.square(np.mean(np.sqrt(np.abs(np_data))))
    statistic_dict['comb_rms'] = np.sqrt(np.mean(np.square(np_data)))
    statistic_dict['comb_rss'] = np.sqrt(np.sum(np_data))
    statistic_dict['comb_peak'] = np.max(np.abs(np_data))
    statistic_dict['comb_p2p'] = np.max(np_data) - np.min(np_data)
    # factor
    statistic_dict['factor_crest'] = statistic_dict['comb_peak'] / statistic_dict['comb_rms']
    statistic_dict['factor_clearance'] = statistic_dict['comb_peak'] / (np.square(np.sum(np.sqrt(np.abs(np_data))))/len(np_data))
    statistic_dict['factor_shape'] = statistic_dict['comb_rms'] / statistic_dict['data_var']
    statistic_dict['factor_impulse'] = statistic_dict['comb_peak'] / statistic_dict['data_var']

    return statistic_dict

def extract_feature_day(root):
    files = os.listdir(root) # 70행 34열 20개 파일
    feature_dict = dict()

    for index_day, day_file in enumerate(files):
        dataset = pd.read_csv(os.path.join(root, day_file))
        numeric_data, label = drop_col(dataset)
        columns = numeric_data.columns.to_list()
        date = day_file.replace('pqm_2_', '').replace('.csv', '').replace('2022', '')
        for column in columns:
            data = dataset[column] # 한 줄씩 뽑음
            features = get_features(data) # 18개의 feature가 나옴 dict
            features['date'] = date
            features['label'] = label
            for key, value in features.items():
                if key not in feature_dict:
                    feature_data = pd.DataFrame(index=range(0, len(files)), columns=columns)
                    feature_dict[key] = feature_data
                if isinstance(value, str):
                    feature_dict[key][column].iloc[index_day] = value
                else:
                    feature_dict[key][column].iloc[index_day] = value.item()

    for key, value in feature_dict.items():
        filename = key + '.csv'
        root = r'D:\MnS\Projects\SerialModel\data\history\04 features'
        value.to_csv(os.path.join(root, filename))







def drop_col(dataset):
    label = dataset['label']
    del dataset['label']
    # 학습 전, 불필요한 컬럼 삭제하기
    drop_filter = ['idx', 'daq_id', 'op', 'type', 'year', 'mon', 'day', 'hour', 'min', 'sec',
                   'total_time', 'weekday', 'time_index', 'month']
    drop_col = []
    # Normalize 시도하고, 시도하는 중에 std가 0인 경우 drop_col에 추가하기
    norm_dataset = copy.deepcopy(dataset)
    for col in dataset.columns:
        pass_flag = False
        for drop in drop_filter:
            if drop in col:
                pass_flag = True
                drop_col.append(col)
                break
        if pass_flag is True:
            continue

        data = dataset[col]
        gap = data.max() - data.min()
        if gap == 0:
            drop_col.append(col)
            continue
        norm_dataset[col] = data

    # 불필요한 컬럼 제거
    for drop in drop_col:
        del norm_dataset[drop]

    return norm_dataset, label[0]