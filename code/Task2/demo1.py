#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[2]:


# 加载训练数据
train_df = pd.read_csv('train.csv')

# 查看前5行数据
print(train_df.head())

# 查看数据基本信息
print("\n数据基本信息:")
print(train_df.info())


# In[3]:


# 检查缺失值(标记为'?')
print("\n缺失值检查:")
# 先检查有多少单元格包含'?'
missing_values = (train_df == '?').sum()
print(missing_values)

# 将'?'替换为NaN以便后续处理
train_df.replace('?', np.nan, inplace=True)

# 现在用常规方法检查NaN
print("\nNaN值统计:")
print(train_df.isnull().sum())


# In[4]:


# 查看标签分布
print("\n标签分布:")
label_dist = train_df['label'].value_counts()
print(label_dist)

# 可视化标签分布
plt.figure(figsize=(8,5))
sns.countplot(x='label', data=train_df)
plt.title('Class Label Distribution')
plt.show()


# In[5]:


# 分离特征和标签
features = train_df.drop(columns=['label', 'id'])  # 假设id列名为'id'
labels = train_df['label']

# 查看特征类型
print("\n特征数据类型:")
print(features.dtypes)

# 对于数值型特征，绘制分布图
numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
if len(numeric_features) > 0:
    features[numeric_features].hist(bins=20, figsize=(15,10))
    plt.tight_layout()
    plt.show()


# In[6]:


# 选择几个特征查看与标签的关系
if len(numeric_features) >= 3:
    sample_features = numeric_features[:3]  # 只看前三个特征为例
    for feature in sample_features:
        plt.figure(figsize=(8,5))
        sns.boxplot(x='label', y=feature, data=train_df)
        plt.title(f'{feature} by Label')
        plt.show()


# In[7]:


# 计算特征间的相关性(只针对数值特征)
if len(numeric_features) > 1:
    corr_matrix = features[numeric_features].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()



# ## Data Processing

# ### (1) 少量缺失特征（X31/Y31/Z31 - 各355个）

# In[8]:


# 转换为数值型并填充中位数（对异常值更鲁棒）
for col in ['X31','Y31','Z31']:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    train_df[col] = train_df[col].fillna(train_df[col].median())


# In[9]:


# 使用KNN填充（考虑特征间关系）
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
cols = ['X41','Y41','Z41']
train_df[cols] = imputer.fit_transform(train_df[cols])


# ### (2) 中等缺失特征（X41/Y41/Z41 - 各1,604个）

# In[10]:


# 使用KNN填充（考虑特征间关系）
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
cols = ['X41','Y41','Z41']
train_df[cols] = imputer.fit_transform(train_df[cols])


# ### (3) 高缺失特征（X51/Y51/Z51 - 各6,634个）

# In[11]:


# 创建缺失标志后删除列（避免噪声）
for col in ['X51','Y51','Z51']:
    train_df[f'{col}_missing'] = train_df[col].isnull().astype(int)
train_df.drop(['X51','Y51','Z51'], axis=1, inplace=True)


# In[12]:


# Convert all objects to float.
object_cols = train_df.select_dtypes(include='object').columns
train_df[object_cols] = train_df[object_cols].apply(pd.to_numeric, errors='coerce')


# ### (4) 异常值处理

# In[13]:


# 异常值缩尾处理
from scipy.stats.mstats import winsorize

numeric_cols = train_df.select_dtypes(include=['float64']).columns
for col in numeric_cols:
    train_df[col] = winsorize(train_df[col], limits=[0.05, 0.05])


# ### (5)特征工程

# In[14]:


# 创建向量模长特征
coord_pairs = [('01','X01','Y01','Z01'), 
               ('11','X11','Y11','Z11'),
               ('21','X21','Y21','Z21')]

for suffix, x, y, z in coord_pairs:
    train_df[f'magnitude_{suffix}'] = np.sqrt(
        train_df[x]**2 + train_df[y]**2 + train_df[z]**2)

# 创建统计特征
# 每组(X,Y,Z)的均值/标准差
for suffix in ['01','11','21']:
    cols = [f'X{suffix}', f'Y{suffix}', f'Z{suffix}']
    train_df[f'mean_{suffix}'] = train_df[cols].mean(axis=1)
    train_df[f'std_{suffix}'] = train_df[cols].std(axis=1)


# ### (6) 特征缩放

# In[15]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_to_scale = [col for col in train_df.columns 
                    if col not in ['id', 'label'] and 'missing' not in col]
train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])


# ### (7) 分类标签处理

# In[16]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label'])


# ### (8) 数据拆分

# In[17]:


# 分离特征和标签
X = train_df.drop(['id', 'label'], axis=1)
y = train_df['label']

# 训练集/验证集划分
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


# In[18]:


# 检查预处理后数据
print(X_train.isnull().sum().sum())  # 应输出0
print(X_train.describe())  # 查看缩放后分布

# 可视化特征相关性
plt.figure(figsize=(15,10))
sns.heatmap(X_train.corr(), cmap='coolwarm')
plt.show()

# df_sklearn_encoded.to_csv('train_new.csv', index=False)


# # ML and GridSearch

# ### Decision Tree

# In[2]:


from randomForest import DecisionTree
from sklearn.metrics import classification_report

# Model Initialization
models = {
    "PyTorch Decision Tree": DecisionTree(max_depth=1, criterion="gini"),
    # "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)  # 可选对比
}

# 快速评估
for name, model in models.items():
    print(f"Training {name}...")  # 修正：仅打印模型名称
    model.fit(X_train, y_train)   # 注意：应使用 X_train 而非 X
    y_pred = model.predict(X_val)
    print(f"\n{name} Performance:")
    print(classification_report(y_val, y_pred.numpy()))  # 注意 .numpy() 转换


# In[19]:


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# 初始化模型
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(objective='multi:softmax', num_class=5, random_state=42)
}

# 快速评估
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"\n{name} Performance:")
    print(classification_report(y_val, y_pred))


# ### XGBoost

# In[20]:


from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  
from imblearn.pipeline import Pipeline  
from scipy.stats import loguniform, randint, uniform
from joblib import load, dump
import os


# In[33]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# # 初始化模型（修复objective参数名）
# xgb_model = XGBClassifier(
#     objective='multi:softmax',  # 多分类任务
#     num_class=5,                # 类别数
#     n_estimators=100,
#     random_state=42
# )

# # 训练与评估
# xgb_model.fit(X_train, y_train)
# y_pred = xgb_model.predict(X_val)
# print(classification_report(y_val, y_pred))

# # 超参数优化 (GridSearch)
# param_grid = {
#     'learning_rate': [0.01, 0.1],
#     'max_depth': [3, 5],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }

# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
# grid.fit(X_train, y_train)
# print("Best params:", grid.best_params_)

xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    n_estimators=100,
    random_state=42
)

# Baseline
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_val)
print("==== Baseline XGBoost ====")
print(classification_report(y_val, y_pred))

# 网格搜索
param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)

print("==== GridSearch Best Params (XGB) ====")
print(grid.best_params_)

xgb_best = grid.best_estimator_
y_pred_best = xgb_best.predict(X_val)
print("==== XGB Best Model on val ====")
print(classification_report(y_val, y_pred_best))

# 保存最佳模型
dump(xgb_best, "saved_models/XGBoost.joblib")
print("XGBoost最佳模型已保存: saved_models/XGBoost.joblib")


# ### Random Forest

# In[31]:


# from sklearn.ensemble import RandomForestClassifier

# rf_model = RandomForestClassifier(
#     n_estimators=200,
#     class_weight='balanced',  # 处理类别不平衡
#     random_state=42
# )

# rf_model.fit(X_train, y_train)
# y_pred = rf_model.predict(X_val)
# print(classification_report(y_val, y_pred))

# # 特征重要性可视化
# import matplotlib.pyplot as plt
# plt.barh(X_train.columns, rf_model.feature_importances_)
# plt.title("Random Forest Feature Importance")
# plt.show()

# # 超参数优化
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5]
# }
# grid = GridSearchCV(rf_model, param_grid, cv=3, scoring='f1_macro')
# grid.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier

# 假设你已有 X_train, y_train, X_val, y_val
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',  # 用于不平衡数据
    random_state=42
)

# 先直接fit，观察baseline
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_val)
print("==== Baseline RF ====")
print(classification_report(y_val, y_pred))

# 网格搜索
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(rf_model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)

# 输出最佳参数
print("==== GridSearch Best Params (RF) ====")
print(grid.best_params_)

# 在验证集上看效果
rf_best = grid.best_estimator_
y_pred_best = rf_best.predict(X_val)
print("==== RF Best Model on val ====")
print(classification_report(y_val, y_pred_best))

# 保存最佳模型
dump(rf_best, "saved_models/Random_Forest.joblib")
print("RF最佳模型已保存: saved_models/Random_Forest.joblib")


# ### Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 逻辑回归需要数据缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lr_model = LogisticRegression(
    multi_class='multinomial',  # 多分类
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_val_scaled)
print(classification_report(y_val, y_pred))

# 超参数优化
param_grid = {
    'C': [0.1, 1, 10],        # 正则化强度
    'penalty': ['l2']          # 多分类任务通常用l2
}
grid = GridSearchCV(lr_model, param_grid, cv=3, scoring='f1_macro')
grid.fit(X_train_scaled, y_train)


# ### SVM

# In[32]:


# from sklearn.svm import SVC

# svm_model = SVC(
#     kernel='rbf',              
#     class_weight='balanced',   
#     probability=True,        
#     random_state=42
# )

# svm_model.fit(X_train_scaled, y_train)
# y_pred = svm_model.predict(X_val_scaled)
# print(classification_report(y_val, y_pred))

# # 超参数优化
# param_grid = {
#     'C': [0.1, 1, 10],
#     'gamma': ['scale', 'auto']
# }
# grid = GridSearchCV(svm_model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
# grid.fit(X_train_scaled, y_train)
from sklearn.svm import SVC

# 示例：如果需要标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

svm_model = SVC(
    kernel='rbf',
    class_weight='balanced',
    probability=True,  # 需要概率输出就加上
    random_state=42
)

# Baseline
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_val_scaled)
print("==== Baseline SVM ====")
print(classification_report(y_val, y_pred))

# 网格搜索
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}
grid = GridSearchCV(svm_model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

print("==== GridSearch Best Params (SVM) ====")
print(grid.best_params_)

svm_best = grid.best_estimator_
y_pred_best = svm_best.predict(X_val_scaled)
print("==== SVM Best Model on val ====")
print(classification_report(y_val, y_pred_best))

# 保存最佳模型
dump(svm_best, "saved_models/SVM.joblib")
print("SVM最佳模型已保存: saved_models/SVM.joblib")

# **同时** 也要保存好训练所用到的Scaler (如果你需要在预测时对test做同样transform)
dump(scaler, "saved_models/SVM_Scaler.joblib")
print("SVM所用的Scaler已保存: saved_models/SVM_Scaler.joblib")


# In[34]:


from sklearn.metrics import f1_score

models = {
    "XGBoost": xgb_model,
    "Random Forest": rf_model,
    "Logistic Regression": lr_model,
    "SVM": svm_model
}

for name, model in models.items():
    if "Logistic" in name or "SVM" in name:
        y_pred = model.predict(X_val_scaled)  # 注意缩放数据
    else:
        y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"{name} Macro-F1: {f1:.4f}")


# ## Test Dataset Prediction and Result Saving

# In[37]:





# In[ ]:




