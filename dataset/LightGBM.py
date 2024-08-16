import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import lightgbm as lgb
from collections import Counter

# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    data = pd.concat(pd.read_csv(file_path, sep='\t', chunksize=100000), axis=0)
    data['Nextstrain_clade'] = data['Nextstrain_clade'].str.strip()
    data = data[data['Nextstrain_clade'] != 'recombinant']
    data['normalized_day'] = MinMaxScaler().fit_transform(data[['day']])
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.to_period('M')
    return data

# Clade 계층 구조 정의
clade_hierarchy = {
    '19B': '19A', '20A': '19A', '20B': '20A', '21A': '20A', '20C': '20A', '21H': '20A', '21D': '20A',
    '21B': '20A', '20E': '20A', '21I': '21A', '21J': '21A', '21F': '20C', '21C': '20C', '20H': '20C',
    '20G': '20C', '21E': '20B', '20J': '20B', '20I': '20B', '20F': '20B', '20D': '20B', '21G': '20D',
    '21M': '20B', '21K': '21M', '21L': '21M', '23I': '21L', '22D': '21L', '23C': '22D', '22C': '21L',
    '22A': '21L', '22B': '21L', '22E': '22B', '22F': '21L', '23A': '22F', '23G': '23A', '23B': '22F',
    '23E': '22F', '23D': '22F', '23F': '23D', '23H': '23F', '24A': '23H', '24B': '23H', '24C': '24A'
}

# 변이 인코딩 함수
def encode_mutations(mutations, start=437, end=508):
    encoded = np.zeros(end - start + 1, dtype=np.int8)
    for mut in str(mutations).split(','):
        if mut and start <= int(mut[1:-1]) <= end:
            encoded[int(mut[1:-1]) - start] = 1
    return encoded

# 데이터 준비 함수
def prepare_data(data, clades_to_include):
    filtered_data = data[data['Nextstrain_clade'].isin(clades_to_include)]
    X = np.array(filtered_data['aaSubstitutions'].apply(lambda x: encode_mutations(x, 437, 508)).tolist())
    y = filtered_data['Nextstrain_clade'].values
    return X, y

# LGBM 모델 학습 함수
def train_lightgbm_model(X_train, y_train, model_path, label_encoder):
    y_train = label_encoder.transform(y_train)
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),
        'device': 'gpu',
        'gpu_device_id': 1,
        'max_depth': 10,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 10,
        'min_sum_hessian_in_leaf': 10
    }
    model = lgb.train(params, train_data)
    joblib.dump(model, model_path)
    return model

# LGBM 모델 평가 함수
def evaluate_model(X_test, y_test, model_path, label_encoder):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test_encoded = label_encoder.transform(y_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='weighted')
    recall = recall_score(y_test_encoded, y_pred, average='weighted')
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# LGBM 모델 예측 함수
def predict_and_save_results(data, model_paths, label_encoder):
    X_24AB = prepare_data(data, ['24A', '24B'])[0]
    prediction_results = []

    for idx, (model_key, model_path) in enumerate(model_paths.items(), start=1):
        model = joblib.load(model_path)
        y_pred = model.predict(X_24AB)
        y_pred = np.argmax(y_pred, axis=1)
        all_combinations = get_all_combinations(y_pred, len(X_24AB))

        for combo, freq in all_combinations:
            prediction_results.append({
                'Model': f'Training dataset{idx}',
                'Clade': '24A/24B',
                'Combination': combo,
                'Frequency': freq
            })

    prediction_df = pd.DataFrame(prediction_results)
    prediction_df.to_csv('lightgbm_prediction_results_24AB.csv', index=False)
    print(prediction_df)

# 변이 조합 계산 함수
def get_all_combinations(predictions, total_samples):
    counter = Counter([tuple(np.where(row == 1)[0] + 437) for row in predictions])
    all_combinations = counter.most_common()
    all_combinations_with_freq = [(', '.join(map(str, combo)), freq / total_samples) for combo, freq in all_combinations]
    return all_combinations_with_freq

# 통합 실행
file_path = 'filtered_data.txt'
data = load_and_preprocess_data(file_path)

# 모델 경로 설정
model_paths = {
    'before_omicron': 'models/Training_dataset1_LGBM.pkl',
    'till_ba1_ba2': 'models/Training_dataset2_LGBM.pkl',
    'till_ba4_ba5': 'models/Training_dataset3_LGBM.pkl',
    'till_xbb_bq': 'models/Training_dataset4_LGBM.pkl',
    'omicron_BA.1/2': 'models/Training_dataset5_LGBM.pkl',
    'omicron-BA.1/2/4/5': 'models/Training_dataset6_LGBM.pkl'
}

label_encoder = LabelEncoder()
label_encoder.fit(data['Nextstrain_clade'])

# 데이터셋 준비
clades_before_omicron = ['19A', '19B', '20A', '20C', '21H', '21D', '21B', '20E', '21F', '21C', '20H', '20G', '21A', '21I', '21J', '21B', '21E', '20J', '20I', '20F', '20D', '21G']
clades_till_ba1_ba2 = clades_before_omicron + ['21M', '21K', '21L']
clades_till_ba4_ba5 = clades_till_ba1_ba2 + ['22D', '22C', '22B', '22A']
clades_till_xbb_bq = clades_till_ba4_ba5 + ['22F', '22E', '23A', '23B']
omicron1 = ['21M', '21K', '21L']
omicron2 = omicron1 + ['22D', '22C', '22B', '22A']

data_sets = {
    'before_omicron': prepare_data(data, clades_before_omicron),
    'till_ba1_ba2': prepare_data(data, clades_till_ba1_ba2),
    'till_ba4_ba5': prepare_data(data, clades_till_ba4_ba5),
    'till_xbb_bq': prepare_data(data, clades_till_xbb_bq),
    'omicron_BA.1/2': prepare_data(data, omicron1),
    'omicron-BA.1/2/4/5': prepare_data(data, omicron2)
}

# 모델 학습 및 평가
performance_results = []

for idx, (key, (X, y)) in enumerate(data_sets.items(), start=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_path = model_paths[key]
    train_lightgbm_model(X_train, y_train, model_path, label_encoder)
    accuracy, precision, recall, f1 = evaluate_model(X_test, y_test, model_path, label_encoder)
    
    performance_results.append({
        'Model': f'Training dataset{idx}',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

performance_df = pd.DataFrame(performance_results)
performance_df.to_csv('lightgbm_performance.csv', index=False)
print(performance_df)

# 예측 수행 및 결과 저장
predict_and_save_results(data, model_paths, label_encoder)

