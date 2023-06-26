import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/MNS/Desktop/SerialModel/data/dataset/dataset.csv", index_col='created_dt')
df_current = df[['current']]
group_size = 24503
group_index = np.arange(len(df_current)) // group_size

# 통계량 계산
group_statistics_mean = df_current.groupby(group_index).mean()
group_statistics_std = df_current.groupby(group_index).std()
group_statistics_quantile25 = df_current.groupby(group_index).quantile(0.25)
group_statistics_var = df_current.groupby(group_index).var()
group_statistics_skew = df_current.groupby(group_index).skew()
group_statistics_quantile75 = df_current.groupby(group_index).quantile(0.75)
window_size = 100
rolling_mean = df_current.rolling(window=window_size).mean()

# 이동 표준편차 계산
rolling_std = df_current.rolling(window=window_size).std()

# 첨도 계산
rolling_kurtosis = df_current.rolling(window=window_size).kurt()

# 지수 평활법 계산
alpha = 0.2
exponential_smoothing = df_current.ewm(alpha=alpha).mean()

# 그래프 출력
fig, axs = plt.subplots(5, 1, figsize=(8, 10))

# axs[0].plot(range(len(group_statistics_mean)), group_statistics_mean['current'], marker='o', linestyle='-', linewidth=2)
# axs[0].set_ylabel('Mean')
#
# axs[1].plot(range(len(group_statistics_std)), group_statistics_std['current'], marker='o', linestyle='-', linewidth=2)
# axs[1].set_ylabel('Standard Deviation')
#
# axs[2].plot(range(len(group_statistics_quantile25)), group_statistics_quantile25['current'], marker='o', linestyle='-', linewidth=2)
# axs[2].set_ylabel('25th Percentile')
#
# axs[3].plot(range(len(group_statistics_var)), group_statistics_var['current'], marker='o', linestyle='-', linewidth=2)
# axs[3].set_ylabel('Variance')
#
# axs[4].plot(range(len(group_statistics_skew)), group_statistics_skew['current'], marker='o', linestyle='-', linewidth=2)
# axs[4].set_ylabel('Skewness')

axs[0].plot(range(len(group_statistics_quantile75)), group_statistics_quantile75['current'], marker='o', linestyle='-', linewidth=2)
axs[0].set_ylabel('75th Percentile')

axs[1].plot(df_current.index, rolling_mean['current'], linestyle='-', linewidth=2)
axs[1].set_ylabel(f'{window_size}-MA')

axs[2].plot(df_current.index, rolling_std['current'], linestyle='-', linewidth=2)
axs[2].set_ylabel(f'{window_size}-MSD')

axs[3].plot(df_current.index, rolling_kurtosis['current'], linestyle='-', linewidth=2)
axs[3].set_ylabel(f'{window_size}-Kurtosis')

axs[4].plot(df_current.index, exponential_smoothing['current'], linestyle='-', linewidth=2)
axs[4].set_ylabel(f'Exponential Smoothing (alpha={alpha})')

plt.xlabel('Group')
plt.suptitle('Statistics of Current for Each Group', y=0.92)
plt.tight_layout()
plt.show()
