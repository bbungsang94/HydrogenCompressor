import copy
import numba
import os
import datetime
import pandas as pd


def separate_month(root):
    dataset = pd.read_csv(os.path.join(root, "active_dataset.csv"))
    month = dataset['month']
    target_index = {6: [], 7: [], 8: []}
    for idx, item in month.items():
        target_index[item].append(idx)

    for target in target_index.keys():
        dump = dataset.iloc[target_index[target], :]
        dump.to_csv(str(target) + "dataset.csv", index=False)
    # 자 움직여볼가..
    test = 1

def check_active_row(**row):
    ret = False
    desc = "NORMAL"
    work_hours = {'Mon': (8, 18), 'Tue': (8, 18), 'Wed': (8, 18),
                  'Thu': (8, 18), 'Fri': (8, 18), 'Sun': (9, 18)}
    holiday = ['06-01', '08-15']
    break_days = ['06-03', '06-09', '07-20', '08-28']
    warning_days = ['06-01', '06-02', '06-04', '06-05', '06-06', '06-07', '06-08',
                    '07-15', '07-16', '07-17', '07-18', '07-19',
                    '08-23', '08-24', '08-25', '08-26', '08-27']

    if row['time_index'] not in holiday:
        if row['weekday'] in work_hours:
            tup = work_hours[row['weekday']]
            if tup[0] <= row['hour'] <= tup[1]:
                if row['time_index'] in break_days:
                    desc = "BREAK"
                elif row['time_index'] in warning_days:
                    desc = "WARNING"
                ret = True

    return ret, desc


def what_day_is_it(date):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day = date.weekday()
    return days[day]


def add_datetime_column(dataset):
    unix_time = dataset['created_dt']
    total_time = []
    md = []
    month = []
    day = []
    hour = []
    min = []
    sec = []
    weekday = []
    select = []
    label = []
    for idx, item in unix_time.items():
        kor_time = datetime.datetime.utcfromtimestamp(item) + datetime.timedelta(hours=9)
        total_time.append(kor_time.strftime("%Y-%m-%d %H:%M:%S"))
        weekday.append(what_day_is_it(kor_time))
        md.append(kor_time.strftime("%m-%d"))
        month.append(kor_time.month)
        if kor_time.month == 8:
            test = 1
        day.append(kor_time.day)
        hour.append(kor_time.hour)
        min.append(kor_time.minute)
        sec.append(kor_time.second)
        check, desc = check_active_row(time_index=kor_time.strftime("%m-%d"),
                                       weekday=what_day_is_it(kor_time), hour=kor_time.hour)
        if check is True:
            select.append(idx)
            label.append(desc)

    dataset['total_time'] = total_time
    dataset['weekday'] = weekday
    dataset['time_index'] = md
    dataset['month'] = month
    dataset['day'] = day
    dataset['hour'] = hour
    dataset['min'] = min
    dataset['sec'] = sec
    dataset.to_csv("dataset.csv", index=False)
    active_dataset = dataset.iloc[select, :]
    active_dataset['label'] = label
    active_dataset.to_csv("active_dataset.csv", index=False)
    return dataset

def merge_dataset(root):
    folders = os.listdir(root)
    merged = None
    for folder in folders:
        files = os.listdir(os.path.join(root, folder))
        for file in files:
            dataset = pd.read_csv(os.path.join(root, folder, file))
            if merged is None:
                merged = dataset
            else:
                merged = pd.concat([merged, dataset])
    new_merged = merged.set_index('created_dt')
    new_merged = new_merged.reset_index()
    return new_merged

def main():
    root = '../data/dataset'
    separate_month(root)
    #dataset = merge_dataset(root)
    #dataset = add_datetime_column(dataset)
    #fast_track()
    print("done")


if __name__ == "__main__":
    main()