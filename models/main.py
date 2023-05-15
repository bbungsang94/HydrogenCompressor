import copy
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import easydict
from autoencoder import LSTMAutoEncoder
from dataset import TagDataset
import torch
import torch.nn.functional as F
from celluloid import Camera
from tqdm import tqdm


# Anomaly Score (KPI 추출)
class Anomaly_Calculator:
    def __init__(self, mean: np.array, std: np.array):
        self.mean = mean
        self.std = std

    def __call__(self, recons_error: np.array):
        x = (recons_error - self.mean)
        return np.matmul(np.matmul(x, self.std), x.T)


def run(args, model, train_loader, test_loader):
    # optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # epoch 설정(반복 수)
    epochs = tqdm(range(args.max_iter // len(train_loader) + 1))
    # 학습
    count = 0
    best_loss = 100000000
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()

        # Train 영역
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="training")
        for i, batch_data in train_iterator:
            if count > args.max_iter:
                return model
            count += 1

            batch_data = batch_data.to(args.device)
            predict_values = model(batch_data)
            loss = model.loss_function(*predict_values)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_iterator.set_postfix({
                "train_loss": float(loss),
            })

        # Test 영역
        model.eval()
        eval_loss = 0
        test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="testing")
        with torch.no_grad():
            for i, batch_data in test_iterator:
                batch_data = batch_data.to(args.device)
                predict_values = model(batch_data)
                loss = model.loss_function(*predict_values)
                eval_loss += loss.mean().item()
                test_iterator.set_postfix({
                    "eval_loss": float(loss),
                })
        eval_loss = eval_loss / len(test_loader)
        epochs.set_postfix({
            "Evaluation Score": float(eval_loss),
        })

        # 종료 조건
        if eval_loss < best_loss:
            best_loss = eval_loss
        else:
            if args.early_stop:
                print('early stop condition   best_loss[{}]  eval_loss[{}]'.format(best_loss, eval_loss))
                return model

    return model


def get_loss_list(args, model, test_loader):
    test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="testing")
    loss_list = []
    with torch.no_grad():
        for i, batch_data in test_iterator:
            batch_data = batch_data.to(args.device)
            predict_values = model(batch_data)

            # MAE loss
            loss = F.l1_loss(predict_values[0], predict_values[1], reduce=False)
            loss = loss.mean(dim=1).cpu().numpy()
            loss_list.append(loss)
    loss_list = np.concatenate(loss_list, axis=0)
    return loss_list


# 시간을 window_size 단위로 자르기
def make_data_idx(dates, window_size=1):
    input_idx = []
    for idx in range(window_size - 1, len(dates)):
        cur_date = dates[idx].to_pydatetime()
        in_date = dates[idx - (window_size - 1)].to_pydatetime()

        _in_period = (cur_date - in_date).days * 24 * 60 + (cur_date - in_date).seconds / 60

        # 인덱스 생성
        if _in_period == (window_size - 1):
            input_idx.append(list(range(idx - window_size + 1, idx + 1)))
    return input_idx

def plot_box(temp_df, columns):
    for var_name in temp_df.columns:
        if var_name in columns:
            continue
        del temp_df[var_name]

    for var_name in [item for item in temp_df.columns if item in columns]:
        temp_df[var_name] = (temp_df[var_name] - temp_df[var_name].min()) / (temp_df[var_name].max() - temp_df[var_name].min())

    plt.boxplot(temp_df, vert=True, labels=temp_df.columns)
    plt.ylim([-0.1, 1.1])
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.xticks(rotation=90)
    plt.show()
    plt.clf()


def plot_sensor(temp_df, save_path='sample.mp4'):
    fig = plt.figure(figsize=(16, 6))
    # Animation 객체 인스턴스화
    camera = Camera(fig)
    ax = fig.add_subplot(111)
    pass_col = ['created_dt', 'date', 'idx', 'daq_id', 'label']
    dates = temp_df['date'].to_list()
    labels = temp_df['label'].to_list()

    for var_name in tqdm([item for item in temp_df.columns if item not in pass_col]):
        # 센서 기준(컬럼) 추출
        ax.plot(dates, temp_df[var_name])
        ax.legend([var_name], loc='upper right')

        temp_start = dates[0]
        temp_date = dates[0]
        temp_label = labels[0]  # 첫번째 label "NORMAL"

        # 시간, 레이블링 쌍 순회하며 이미지 갱신
        for xc, value in zip(dates, labels):
            if temp_label != value:
                if temp_label == "WARNING":
                    ax.axvspan(temp_start, temp_date, alpha=0.2, color='blue')
                elif temp_label == "BROKEN":
                    ax.axvspan(temp_start, temp_date, alpha=0.2, color='orange')
                temp_start = xc
                temp_label = value
            # 시간 증가
            temp_date = xc

        camera.snap()

    animation = camera.animate(3000, blit=True)
    animation.save(save_path)


def make_processable_dataset(root, filename):
    columns = {'w', 'va', 'current_unbalance', 'current', 'var', 'pf_average', 'op'
               'r_v', 'r_i', 'r_w', 'r_var', 'r_va', 'r_volt_unbalance', 'r_current_unbalance', 'r_phase', 'r_power_factor', 'r_power_thd',
               's_v', 's_i', 's_w', 's_var', 's_va', 's_volt_unbalance', 's_current_unbalance', 's_phase', 's_power_factor', 's_power_thd',
               't_v', 't_i', 't_w', 't_var', 't_va', 't_volt_unbalance', 't_current_unbalance', 't_phase', 't_power_factor', 't_power_thd',
               'label'}
    del_columns = []
    pass_col = ['created_dt', 'time_index', 'label']

    dataset = pd.read_csv(os.path.join(root, filename))

    for var_name in dataset.columns:
        if var_name not in columns:
            del_columns.append(var_name)
            continue
        if var_name in pass_col:
            continue
        # 센서 기준(컬럼) 추출
        dataset[var_name] = (dataset[var_name] - dataset[var_name].min()) / (dataset[var_name].max() - dataset[var_name].min())
        # value = dataset[var_name].sum()
        # if abs(value) < 0.1:
        #     del_columns.add(var_name)

    for col in del_columns:
        if col in dataset.columns:
            del dataset[col]

    dataset.to_csv(os.path.join(root, 'model_dataset.csv'))


def main(root, filename):
    whole_dataset = pd.read_csv(os.path.join(root, filename), index_col=0)
    whole_dataset.head()

    # Timestamp 기준으로 정렬
    whole_dataset = whole_dataset.set_index('created_dt')
    whole_dataset = whole_dataset.reset_index()

    # 결측치 확인
    (whole_dataset.isnull().sum() / len(whole_dataset)).plot.bar(figsize=(18, 8), colormap='Paired')

    # 데이터 레이블 체크
    whole_dataset['label'].unique()

    # 정상 / 비정상 데이터셋 분류
    normal_df = whole_dataset[whole_dataset['label'] == 'NORMAL']
    abnormal_df = whole_dataset[whole_dataset['label'] != 'NORMAL']

    # 정상 데이터를 학습(7)/ 설정(1) / 검증(1) / 실험(1)용으로서 분리
    interval_n = int(len(normal_df) / 10)
    normal_df1 = normal_df.iloc[0:interval_n * 7]
    normal_df2 = normal_df.iloc[interval_n * 7:interval_n * 8]
    normal_df3 = normal_df.iloc[interval_n * 8:interval_n * 9]
    normal_df4 = normal_df.iloc[interval_n * 9:]

    # 비정상 데이터를 검증(5) / 실험 (5)용으로서 분리
    interval_ab = int(len(abnormal_df) / 2)
    abnormal_df1 = abnormal_df.iloc[0:interval_ab]
    abnormal_df2 = abnormal_df.iloc[interval_ab:]

    # 데이터 정규화를 위하여 분산 및 평균 추출
    mean_df = normal_df.mean()
    std_df = normal_df.std()
    # 계산 목적으로 timestamp는 제외
    del mean_df['created_dt']
    del mean_df['time_index']
    del std_df['created_dt']
    del std_df['time_index']

    # 설정 폴더
    args = easydict.EasyDict({
        "batch_size": 32,  # 배치 사이즈 설정
        "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),  # GPU 사용 여부 설정
        "input_size": 36,  # 입력 차원 설정
        "latent_size": 10,  # Hidden 차원 설정
        "output_size": 36,  # 출력 차원 설정
        "window_size": 10,  # sequence Lenght
        "num_layers": 2,  # LSTM layer 갯수 설정
        "learning_rate": 0.001,  # learning rate 설정
        "max_iter": 100000,  # 총 반복 횟수 설정
        'early_stop': True,  # valid loss가 작아지지 않으면 early stop 조건 설정
    })

    # 데이터셋 클래스 변환
    normal_dataset1 = TagDataset(df=normal_df1, input_size=args.input_size, window_size=args.window_size,
                                 mean_df=mean_df, std_df=std_df)
    normal_dataset2 = TagDataset(df=normal_df2, input_size=args.input_size, window_size=args.window_size,
                                 mean_df=mean_df, std_df=std_df)
    normal_dataset3 = TagDataset(df=normal_df3, input_size=args.input_size, window_size=args.window_size,
                                 mean_df=mean_df, std_df=std_df)
    normal_dataset4 = TagDataset(df=normal_df4, input_size=args.input_size, window_size=args.window_size,
                                 mean_df=mean_df, std_df=std_df)
    abnormal_dataset1 = TagDataset(df=abnormal_df1, input_size=args.input_size, window_size=args.window_size,
                                   mean_df=mean_df, std_df=std_df)
    abnormal_dataset2 = TagDataset(df=abnormal_df2, input_size=args.input_size, window_size=args.window_size,
                                   mean_df=mean_df, std_df=std_df)

    # Data Loader 선언
    train_loader = torch.utils.data.DataLoader(
        dataset=normal_dataset1,
        batch_size=args.batch_size,
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset=normal_dataset2,
        batch_size=args.batch_size,
        shuffle=False)

    # 모델 생성
    model = LSTMAutoEncoder(input_dim=args.input_size,
                            latent_dim=args.latent_size,
                            window_size=args.window_size,
                            num_layers=args.num_layers)
    model.to(args.device)

    # 학습 시작
    model = run(args, model, train_loader, valid_loader)

    # Validation Loss 연산
    loss_list = get_loss_list(args, model, valid_loader)

    # Reconstruction Error의 평균과 Covarinace 계산
    mean = np.mean(loss_list, axis=0)
    std = np.cov(loss_list.T)

    # 비정상 스코어 추출
    anomaly_calculator = Anomaly_Calculator(mean, std)

    # 기준점 탐색
    anomaly_scores = []
    for temp_loss in tqdm(loss_list):
        temp_score = anomaly_calculator(temp_loss)
        anomaly_scores.append(temp_score)

    # 정상구간에서 비정상 점수 분포
    print("평균[{}], 중간[{}], 최소[{}], 최대[{}]".format(np.mean(anomaly_scores), np.median(anomaly_scores),
                                                  np.min(anomaly_scores), np.max(anomaly_scores)))

    anomaly_calculator = Anomaly_Calculator(mean, std)

    # 전체 데이터 불러오기
    total_dataset = TagDataset(df=whole_dataset, input_size=args.input_size, window_size=args.window_size,
                               mean_df=mean_df, std_df=std_df)
    total_dataloader = torch.utils.data.DataLoader(dataset=total_dataset, batch_size=args.batch_size, shuffle=False)

    # Reconstruction Loss를 계산하기
    total_loss = get_loss_list(args, model, total_dataloader)

    # 이상치 점수 계산하기
    anomaly_scores = []
    for temp_loss in tqdm(total_loss):
        temp_score = anomaly_calculator(temp_loss)
        anomaly_scores.append(temp_score)

    visualization_df = total_dataset.df
    visualization_df['score'] = anomaly_scores
    visualization_df['recons_error'] = total_loss.sum(axis=1)

    # 시각화 객체 선언
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111)

    # 불량 구간 탐색 데이터
    labels = visualization_df['label'].values.tolist()
    dates = visualization_df['date'].tolist()

    ax.plot(dates, visualization_df['score'])
    ax.legend(['abnormal score'], loc='upper right')

    # 고장구간 표시
    temp_start = dates[0]
    temp_date = dates[0]
    temp_label = labels[0]

    for xc, value in zip(dates, labels):
        if temp_label != value:
            if temp_label == "WARNING":
                ax.axvspan(temp_start, temp_date, alpha=0.2, color='blue')
            elif temp_label == "BROKEN":
                ax.axvspan(temp_start, temp_date, alpha=0.2, color='orange')
            temp_start = xc
            temp_label = value
        temp_date = xc

    plt.show()
    plt.clf()
    ## 불량 구간 탐색 데이터
    labels = visualization_df['label'].values.tolist()
    dates = visualization_df['date'].tolist()
    y_values = visualization_df['recons_error'].to_list()
    cnt = 0
    for i in range(3):
        xtics = []
        ytics = []
        month_label = []
        months = [date for date in dates if date < (7 + i) * 100]
        for d in months:
            xtics.append(d)
            if y_values[cnt] > 60.0:
                value = 1.0
            else:
                value = 0.0
            #ytics.append(value)
            ytics.append(y_values[cnt])
            month_label.append(copy.deepcopy(labels[cnt]))
            cnt += 1
        ## 시각화 하기
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111)

        ax.plot(xtics, ytics)
        ax.legend(['abnormal error'], loc='upper right')

        ## 고장구간 표시
        temp_start = dates[0]
        temp_date = dates[0]
        temp_label = labels[0]

        for xc, value in zip(xtics, month_label):
            if temp_label != value:
                if temp_label == "WARNING":
                    ax.axvspan(temp_start, temp_date, alpha=0.2, color='blue')
                if temp_label == "BROKEN":
                    ax.axvspan(temp_start, temp_date, alpha=0.2, color='orange')
                temp_start = xc
                temp_label = value
            temp_date = xc
        plt.show()
        plt.clf()


if __name__ == '__main__':
    # make_processable_dataset(root=root_dir, filename='active_dataset.csv')
    main(root=root_dir, filename='model_dataset.csv')
