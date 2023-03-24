import os
import datetime
import pandas as pd


def get_only_operation(root, filename):
    dataset = pd.read_csv(os.path.join(root, filename))
    dataset = dataset[dataset['op'] == 1]
    return dataset


def separate_days(root, filename):
    days = list(range(1, 32))
    dataset = pd.read_csv(os.path.join(root, filename))
    for day in days:
        dump = dataset[dataset['day'] == day]
        if (len(dump) > 1):
            dump.to_csv(os.path.join(root, str(day) + '.csv'), index=False)

def separate_month(root):
    dataset = pd.read_csv(os.path.join(root, "active_dataset.csv"))
    target_index = [6, 7, 8]

    for target in target_index:
        dump = dataset[dataset['month'] == target]
        dump.to_csv(str(target) + "-dataset.csv", index=False)


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

def merge_and_normalize(root):
    files = os.listdir(root)
    merged = None
    for file in files:
        dataset = pd.read_csv(os.path.join(root, file))
        if merged is None:
            merged = dataset
        else:
            merged = pd.concat([merged, dataset])

    # normalize
    """
    created_dt	idx	daq_id	leakage_current
    w	va	current_unbalance
    kwh_sum	kwh_thismonth	kwh_lastmonth	type
    current	var	pf_average	kvarh_sum	kvarh_thismonth
    kvarh_lastmonth	r_v	r_i
    r_w	r_var	r_va
    r_volt_unbalance	r_current_unbalance	r_phase
    r_power_factor	r_power_thd	s_v	s_i	s_w	s_var	s_va	s_volt_unbalance	s_current_unbalance	s_phase	s_power_factor	s_power_thd	t_v	t_i	t_w	t_var	t_va	t_volt_unbalance	t_current_unbalance	t_phase	t_power_factor	t_power_thd	r_swell	s_swell	t_swell	r_sag	s_sag	t_sag	op	r_oc	s_oc	t_oc	sag_year	sag_mon	sag_day	sag_hour	sag_min	sag_sec	swell_year	swell_mon	swell_day	swell_hour	swell_min	swell_sec	total_time	weekday	time_index	month	day	hour	min	sec	label

    """

def main():
    #root = '../data/history'
    root = './'
    month = ['6-', '7-', '8-']
    for mon in month:
        in_root = os.path.join(root, mon)
        separate_days(in_root, mon + 'dataset.csv')
    #separate_month(root)
    # for mon in month:
    #     filename = mon + 'dataset.csv'
    #     dataset = get_only_operation(root, filename)
    #     dataset.to_csv('only_op-' + filename, index=False)
    #separate_month(root)
    #dataset = merge_dataset(root)
    #dataset = add_datetime_column(dataset)
    #fast_track()
    print("done")


if __name__ == "__main__":
    main()