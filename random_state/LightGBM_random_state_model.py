import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# GPU 설정 및 데이터 로드/전처리
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data = pd.concat(pd.read_csv('filtered_data.txt', sep='\t', chunksize=100000), axis=0)
data = data[data['Nextstrain_clade'].str.strip() != 'recombinant'].drop(columns=['deletions', 'insertions'])

# 변이 인코딩 및 라벨 변환 함수
def encode_mutations(mutations, start=437, end=508):
    encoded = np.zeros(end - start + 1, dtype=np.int8)
    for mut in str(mutations).split(','):
        if mut and start <= int(mut[1:-1]) <= end:
            encoded[int(mut[1:-1]) - start] = 1
    return encoded

def encode_labels(X):
    return np.array([np.argmax(row) if np.sum(row) > 0 else -1 for row in X])

# 데이터 준비 및 전처리
X = np.array(data['aaSubstitutions'].apply(lambda x: encode_mutations(x, 437, 508)).tolist())
y = encode_labels(X)
X, y = X[y != -1], y[y != -1]

# 데이터 분할 및 LightGBM 모델 학습
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = lgb.train(
    {'objective': 'multiclass', 'num_class': 72, 'device': 'gpu', 'gpu_device_id': 1},
    lgb.Dataset(X_train, label=y_train),
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# 모델 저장
model.save_model('models/LightGBM_random_state_model.txt')

