import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    data = pd.concat(pd.read_csv(file_path, sep='\t', chunksize=100000), axis=0)
    data['Nextstrain_clade'] = data['Nextstrain_clade'].str.strip()
    data = data[data['Nextstrain_clade'] != 'recombinant']
    data['normalized_day'] = MinMaxScaler().fit_transform(data[['day']])
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.to_period('M')
    return data

# 변이 인코딩 함수
def encode_mutations(mutations, start=437, end=508):
    encoded = np.zeros(end - start + 1, dtype=np.int8)
    for mut in str(mutations).split(','):
        if mut and start <= int(mut[1:-1]) <= end:
            encoded[int(mut[1:-1]) - start] = 1
    return encoded

# Clade 계층 구조 정의
clade_hierarchy = {
    '19B': '19A', '20A': '19A', '20B': '20A', '21A': '20A', '20C': '20A', '21H': '20A', '21D': '20A',
    '21B': '20A', '20E': '20A', '21I': '21A', '21J': '21A', '21F': '20C', '21C': '20C', '20H': '20C',
    '20G': '20C', '21E': '20B', '20J': '20B', '20I': '20B', '20F': '20B', '20D': '20B', '21G': '20D',
    '21M': '20B', '21K': '21M', '21L': '21M', '23I': '21L', '22D': '21L', '23C': '22D', '22C': '21L',
    '22A': '21L', '22B': '21L', '22E': '22B', '22F': '21L', '23A': '22F', '23G': '23A', '23B': '22F',
    '23E': '22F', '23D': '22F', '23F': '23D', '23H': '23F', '24A': '23H', '24B': '23H', '24C': '24A'
}

# 데이터 준비 함수
def prepare_data(data, clades_to_include):
    filtered_data = data[data['Nextstrain_clade'].isin(clades_to_include)]
    if filtered_data.empty:
        print(f"No data found for clades: {clades_to_include}")
        return np.array([]), np.array([])
    X = np.array(filtered_data['aaSubstitutions'].apply(lambda x: encode_mutations(x, 437, 508)).tolist())
    y = filtered_data['Nextstrain_clade'].values
    return X, y

# Random Forest 모델 학습 및 평가
def train_and_evaluate_model(data, model_paths, clade_hierarchy, label_encoder):
    performance_results = []

    for idx, (key, clades) in enumerate(model_paths.items(), start=1):
        X, y = prepare_data(data, clades)
        if X.size == 0:
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_paths[key])

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        performance_results.append({
            'Model': f'Training dataset{idx}',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    performance_df = pd.DataFrame(performance_results)
    performance_df.to_csv('Random_Forest_performance.csv', index=False)
    print(performance_df)

# 예측 수행 및 결과 저장
def predict_and_save_results(data, model_paths, clade_hierarchy):
    prediction_results = []
    data_sets = {
        '24A/24B': prepare_data(data, ['24A', '24B'])
    }

    for data_key, (X, _) in data_sets.items():
        if X.size == 0:
            continue

        for idx, (model_key, model_path) in enumerate(model_paths.items(), start=1):
            y_pred = predict_mutations(X, model_path)
            all_combinations = get_all_combinations(y_pred, len(X))

            for combo, freq in all_combinations:
                prediction_results.append({
                    'Input Clade': data_key,
                    'Model': f'Training dataset{idx}',
                    'Combination': combo,
                    'Frequency': freq
                })

    prediction_df = pd.DataFrame(prediction_results)
    prediction_df.to_csv('Random_Forest_prediction_results_24AB.csv', sep='\t', index=False)
    print(prediction_df)

def predict_mutations(X_test, model_path):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    return y_pred

def get_all_combinations(predictions, total_samples):
    mutation_positions = np.arange(437, 509)
    decoded_mutations = [','.join([str(mutation_positions[idx]) for idx in np.where(pred == 1)[0]]) for pred in predictions]
    counts = Counter(decoded_mutations)
    all_combinations_percentage = [(combo, count / total_samples) for combo, count in counts.items()]
    return all_combinations_percentage

# 통합 실행
file_path = 'filtered_data.txt'
data = load_and_preprocess_data(file_path)

# 모델 경로 설정
model_paths = {
    'before_omicron': 'models/Training_dataset1_RF.pkl',
    'till_ba1_ba2': 'models/Training_dataset2_RF.pkl',
    'till_ba4_ba5': 'models/Training_dataset3_RF.pkl',
    'till_xbb_bq': 'models/Training_dataset4_RF.pkl',
    'omicron_BA.1/2': 'models/Training_dataset5_RF.pkl',
    'omicron-BA.1/2/4/5': 'models/Training_dataset6_RF.pkl'
}

label_encoder = LabelEncoder()

train_and_evaluate_model(data, model_paths, clade_hierarchy, label_encoder)
predict_and_save_results(data, model_paths, clade_hierarchy)
