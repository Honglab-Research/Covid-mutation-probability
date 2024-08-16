import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
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

# XGBoost 모델 학습
def train_xgboost_model(X_train, y_train, model_path):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', tree_method='gpu_hist', gpu_id=0)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

# 모델 경로 설정
model_paths = {
    'Training dataset1': 'models/Training_dataset1_XGBoost.pkl',
    'Training dataset2': 'models/Training_dataset2_XGBoost.pkl',
    'Training dataset3': 'models/Training_dataset3_XGBoost.pkl',
    'Training dataset4': 'models/Training_dataset4_XGBoost.pkl',
    'Training dataset5': 'models/Training_dataset5_XGBoost.pkl',
    'Training dataset6': 'models/Training_dataset6_XGBoost.pkl'
}

# 데이터셋 정의
clades_before_omicron = [
    '19A', '19B', '20A', '20C', '21H', '21D', '21B', '20E', '21F', '21C',
    '20H', '20G', '21A', '21I', '21J', '21B', '21E', '20J', '20I', '20F',
    '20D', '21G'
]

clades_till_ba1_ba2 = clades_before_omicron + [
    '21M', '21K', '21L'
]

clades_till_ba4_ba5 = clades_till_ba1_ba2 + [
    '22D', '22C', '22B', '22A'
]

clades_till_xbb_bq = clades_till_ba4_ba5 + [
    '22F', '22E', '23A', '23B'
]

omicron1 = ['21M', '21K', '21L']
omicron2 = omicron1 + ['22D', '22C', '22B', '22A']

data_sets = {
    'before_omicron': prepare_data(load_and_preprocess_data('/data/git/230522/240802/aa_fre/extracted_data.txt'), clades_before_omicron),
    'till_ba1_ba2': prepare_data(load_and_preprocess_data('/data/git/230522/240802/aa_fre/extracted_data.txt'), clades_till_ba1_ba2),
    'till_ba4_ba5': prepare_data(load_and_preprocess_data('/data/git/230522/240802/aa_fre/extracted_data.txt'), clades_till_ba4_ba5),
    'till_xbb_bq': prepare_data(load_and_preprocess_data('/data/git/230522/240802/aa_fre/extracted_data.txt'), clades_till_xbb_bq),
    'omicron_BA.1/2': prepare_data(load_and_preprocess_data('/data/git/230522/240802/aa_fre/extracted_data.txt'), omicron1),
    'omicron-BA.1/2/4/5': prepare_data(load_and_preprocess_data('/data/git/230522/240802/aa_fre/extracted_data.txt'), omicron2)
}

# 학습 실행
for idx, (key, (X, y)) in enumerate(data_sets.items(), start=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_path = model_paths[f'Training dataset{idx}']
    train_xgboost_model(X_train, y_train, model_path)

# XGBoost 예측 및 결과 저장
def predict_and_save_results(X_test, model_paths):
    prediction_results = []
    for idx, (model_key, model_path) in enumerate(model_paths.items(), start=1):
        print(f"Predicting with model: {model_key}")
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        all_combinations = get_all_combinations(y_pred, len(X_test))

        for combo, freq in all_combinations:
            prediction_results.append({
                'Model': f'Training dataset{idx}',
                'Combination': combo,
                'Frequency': freq
            })

    prediction_df = pd.DataFrame(prediction_results)
    prediction_df.to_csv('xgboost_prediction_results_24AB.csv', index=False)
    print(prediction_df)

# 모든 조합 및 빈도 계산 함수
def get_all_combinations(predictions, total_samples):
    mutation_positions = np.arange(437, 509)
    decoded_mutations = [','.join([str(mutation_positions[idx]) for idx in np.where(pred == 1)[0]]) for pred in predictions]
    counts = Counter(decoded_mutations)
    all_combinations_percentage = [(combo, count / total_samples) for combo, count in counts.items()]
    return all_combinations_percentage

# 데이터셋 준비 (예측용)
X_24AB, _ = prepare_data(load_and_preprocess_data('filtered_data.txt'), ['24A', '24B'])

# 예측 실행 및 결과 저장
predict_and_save_results(X_24AB, model_paths)

