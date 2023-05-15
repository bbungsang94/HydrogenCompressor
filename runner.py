import os
import copy
import numpy as np
import pandas as pd
import easydict
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datetime
from tqdm import tqdm


class Anomaly_Calculator:
    def __init__(self, mean: np.array, std: np.array):
        self.mean = mean
        self.std = std

    def __call__(self, recons_error: np.array):
        x = (recons_error - self.mean)
        return np.matmul(np.matmul(x, self.std), x.T)


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
        norm_dataset[col] = (data - data.min()) / gap

    # 불필요한 컬럼 본격적으로 제거
    for drop in drop_col:
        del norm_dataset[drop]

    norm_dataset['label'] = label
    return norm_dataset


def patch_label_on_pca(path='./data/history/06 PCA/dataset.csv'):
    dataset = pd.read_csv(path, index_col='created_dt')
    label_data = pd.read_csv(r'D:\MnS\Projects\SerialModel\data\history\05 dataset for train\dataset.csv', index_col='created_dt')
    dataset['label'] = label_data['label']
    print(dataset.head(10))
    dataset.to_csv('./data/history/06 PCA/dataset_label.csv')


def train_all_process(root, filename):
    from models.dataset import TagDataset
 ##   from models.autoencoder import LSTMAutoEncoder
    from models.IsoForest_Autoencoder import LSTMAutoEncoder

    whole_dataset = pd.read_csv(os.path.join(root, filename), index_col='created_dt')
    # 결측치 확인

    (whole_dataset.isnull().sum() / len(whole_dataset)).plot.bar(figsize=(18, 8), colormap='Paired')
    # 데이터 레이블 체크
    print(whole_dataset['label'].unique())
    # 정상 / 비정상 데이터셋 분류
    normal_df = whole_dataset[whole_dataset['label'] == 'NORMAL']
    abnormal_df = whole_dataset[whole_dataset['label'] != 'NORMAL']
    del normal_df['label']
    del abnormal_df['label']

    # 정상 데이터를 학습(4)/ 검증(3) / 실험(3)용으로서 분리
    interval_n = len(normal_df) // 10
    normal_train = normal_df.iloc[0:interval_n * 4]
    normal_val= normal_df.iloc[interval_n * 4:interval_n * 7]
    normal_test = normal_df.iloc[interval_n * 7:]

    # 비정상 데이터를 검증(5) / 실험 (5)용으로서 분리
    interval_ab = len(abnormal_df) // 2
    abnormal_val = abnormal_df.iloc[0:interval_ab]
    abnormal_test = abnormal_df.iloc[interval_ab:]

    # 데이터 정규화를 위하여 분산 및 평균 추출
    if os.path.isfile(os.path.join(root, 'mean_df.csv')) is False:
        mean_df = normal_df.mean()
        std_df = normal_df.std()
        std_df.T.to_csv('std_df.csv')
        mean_df.T.to_csv('mean_df.csv')
    else:
        std_df = pd.read_csv(os.path.join(root, 'std_df.csv'))
        mean_df = pd.read_csv(os.path.join(root, 'mean_df.csv'))
        std_df = std_df.squeeze()
        mean_df = mean_df.squeeze()
    args = easydict.EasyDict({
        "batch_size": 32,  # 배치 사이즈 설정
        "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),  # GPU 사용 여부 설정
        "input_size": len(mean_df),  # 입력 차원 설정 38
        "latent_size": 10,  # Hidden 차원 설정
        "output_size": len(mean_df),  # 출력 차원 설정 38
        "window_size": 60,  # sequence Length
        "num_layers": 2,  # LSTM layer 갯수 설정
        "learning_rate": 0.001,  # learning rate 설정
        "max_iter": 100000,  # 총 반복 횟수 설정
        'early_stop': True,  # valid loss가 작아지지 않으면 early stop 조건 설정
    })

    dataset_train = TagDataset(df=normal_train, input_size=args.input_size, window_size=args.window_size,
                               mean_df=mean_df, std_df=std_df)
    dataset_valid = TagDataset(df=normal_val, input_size=args.input_size, window_size=args.window_size,
                               mean_df=mean_df, std_df=std_df)

    # Data Loader 선언
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=args.batch_size,
        shuffle=False)

    # 모델 생성
    model = LSTMAutoEncoder(input_dim=args.input_size,
                            latent_dim=args.latent_size,
                            window_size=args.window_size,
                            num_layers=args.num_layers)
    if os.path.isfile(os.path.join(root, 'data/dataset/LSTM_MODEL.pth')) is True:
        model.load_state_dict(torch.load(os.path.join(root, "data/dataset/LSTM_MODEL.pth")))
        model.eval()
        model.to(args.device)
    else:
        model.to(args.device)
        model = run(args, model, train_loader, valid_loader)
        torch.save(model.state_dict(), 'data/dataset/LSTM_MODEL.pth')

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

    # anomaly_calculator = Anomaly_Calculator(mean, std)

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
    visualization_df.to_csv('result.csv')
    # 시각화 객체 선언
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111)

    # 불량 구간 탐색 데이터
    labels = visualization_df['label'].values.tolist()
    dates = visualization_df.index.tolist()

    ax.scatter(dates, visualization_df['score'])
    # ax.plot(dates, visualization_df['score'])
    ax.legend(['abnormal score'], loc='upper right')

    # 고장구간 표시
    temp_start = dates[0]
    temp_date = dates[0]
    temp_label = labels[0]

    for xc, value in zip(dates, labels):
        if temp_label != value:
            if temp_label == "WARNING":
                ax.axvspan(temp_start, temp_date, alpha=0.2, color='orange')
            elif temp_label == "BREAK":
                ax.axvspan(temp_start, temp_date, alpha=0.2, color='red')
            temp_start = xc
            temp_label = value
        temp_date = xc

    plt.show()
    plt.clf()
    ## 불량 구간 탐색 데이터
    labels = visualization_df['label'].values.tolist()
    dates = visualization_df.index.tolist()
    y_values = visualization_df['recons_error'].to_list()
    cnt = 0
    # kor_time = datetime.datetime.utcfromtimestamp(item) + datetime.timedelta(hours=9)
    for i in range(6,9):
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
                    ax.axvspan(temp_start, temp_date, alpha=0.2, color='orange')
                if temp_label == "BREAK":
                    ax.axvspan(temp_start, temp_date, alpha=0.2, color='red')
                temp_start = xc
                temp_label = value
            temp_date = xc
        plt.show()
        plt.clf()
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


def main():
    history_path = r'C:\Users\MNS\Desktop\SerialModel\data\history'
    source_path = r'C:\Users\MNS\Desktop\SerialModel\data\dataset'
    """
    #region source 업무시간제거 및 레이블링
    from utils.files import add_datetime_column

    folders = os.listdir(source_path)
    for folder in folders:
        files = os.listdir(os.path.join(source_path, folder))
        for day_dataset in files:
            dataset = pd.read_csv(os.path.join(source_path, folder, day_dataset))
            dataset = dataset.reset_index()
            dataset = add_datetime_column(dataset)
            if len(dataset) > 0:
                dataset.to_csv(os.path.join(history_path, day_dataset), index=False)
    #endregion
    """

    """
    #region swell, oc, sag 자르기
    from utils.files import del_swell_oc_sag
    files = os.listdir(source_path)
    for day_dataset in files:
        dataset = pd.read_csv(os.path.join(source_path, day_dataset))
        dataset = del_swell_oc_sag(dataset)
        if len(dataset) > 0:
            dataset.to_csv(os.path.join(history_path, day_dataset), index=False)
    #endregion
    """

    """
    #region merge, drop column and normalize
    from utils.files import merge_dataset
    dataset = merge_dataset(source_path)
    dataset = drop_col(dataset)
    dataset.to_csv(os.path.join(history_path, 'dataset.csv'))
    #endregion
    """

    train_all_process(root=source_path, filename='dataset.csv')


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()