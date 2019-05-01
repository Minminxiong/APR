import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# =============================================================================
# def get_X_y(file):
#     df_train = pd.read_csv(file, sep=",")
#     X_train = df_train.iloc[:,:-2]
#     y_train = df_train.cascade_unique_size
#     return X_train, y_train
# 
# 
# X_train, y_train = get_X_y("features/imeline.train.righ_cascade_features_list_25.csv")
# 
# X_test, y_test = get_X_y("features/imeline.validation.righ_cascade_features_list_25.csv")
# =============================================================================
df_train = pd.read_csv("features/train_features_k25_cas50.csv")

print(df_train.describe())
#print(X_test.describe())
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)

#设置与目标”cascade_unique_size最紧密关联的变量数
k = 10
cols = corrmat.nlargest(k, 'cascade_unique_size')["cascade_unique_size"].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.savefig("cascade_size_features.pdf")
plt.show()
# =============================================================================
# #保存图片
# fig = plt.gcf()
# plt.margins()
# =============================================================================



####################对目标节点预测相关特征分析
df_target_train = pd.read_csv("features/labelmodify_target_features_25.csv")
target_corrmat = df_target_train.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(target_corrmat, vmax=.8, square=True)

k = 10
target_cols = target_corrmat.nlargest(k, "label")["label"].index
target_cm = np.corrcoef(df_target_train[target_cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(target_cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.savefig("target_features.pdf")
plt.show()