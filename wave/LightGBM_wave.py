import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# 데이터 로드 및 전처리
file_path = 'filtered_data.txt'
chunks = []
for chunk in pd.read_csv(file_path, sep='\t', chunksize=100000):
    chunks.append(chunk)
data = pd.concat(chunks, axis=0)
data['Nextstrain_clade'] = data['Nextstrain_clade'].str.strip()
data = data[data['Nextstrain_clade'] != 'recombinant']
scaler = MinMaxScaler()
data['normalized_day'] = scaler.fit_transform(data[['day']])
data['Nextstrain_clade'] = data['Nextstrain_clade'].str.strip()
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.to_period('M')

# 변이 인코딩 함수
def encode_mutations(mutations, start=437, end=508):
    length = end - start + 1
    encoded = np.zeros(length, dtype=np.int8)
    for mut in str(mutations).split(','):
        if mut:
            try:
                pos = int(mut[1:-1])
                if start <= pos <= end:
                    encoded[pos - start] = 1
            except ValueError:
                continue
    return encoded

# 데이터 준비
def prepare_data(subset_data):
    X = np.array(subset_data['aaSubstitutions'].apply(lambda x: encode_mutations(x, 437, 508)).tolist())
    return X

# 변이 위치가 없는 경우 (-1 라벨) 제거
def remove_no_mutations(X, y):
    X, y = zip(*[(x, label) for x, label in zip(X, y) if np.sum(x) > 0])
    return np.array(X), np.array(y)

# LightGBM 모델 학습
def train_lgb_model(X_train, y_train, model_path):
    models = []
    for i in range(y_train.shape[1]):
        d_train = lgb.Dataset(X_train, label=y_train[:, i])

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'device': 'gpu',
            'gpu_device_id': 1
        }

        model = lgb.train(params, d_train, num_boost_round=100)
        models.append(model)

    for i, model in enumerate(models):
        model.save_model(f"{model_path}_{i}.txt")
    print(f"LightGBM models saved to {model_path}_*.txt")

# 모델 성능 평가 및 모든 조합 출력
def evaluate_model(X_input, X_target, model_path, model_name):
    y_pred_labels = np.zeros_like(X_input)
    for i in range(X_input.shape[1]):
        model = lgb.Booster(model_file=f"{model_path}_{i}.txt")
        y_pred = model.predict(X_input)
        y_pred_labels[:, i] = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(X_target[:len(y_pred_labels)], y_pred_labels)
    precision = precision_score(X_target[:len(y_pred_labels)], y_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(X_target[:len(y_pred_labels)], y_pred_labels, average='weighted')
    f1 = f1_score(X_target[:len(y_pred_labels)], y_pred_labels, average='weighted')

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1 Score: {f1:.4f}")

    total_predictions = len(y_pred_labels)
    pred_counter = Counter([tuple(np.where(row == 1)[0] + 437) for row in y_pred_labels])

    all_combinations = [(combo, count / total_predictions) for combo, count in pred_counter.items()]
    all_combinations = sorted(all_combinations, key=lambda x: x[1], reverse=True)
    print(f"{model_name} All Mutations Combinations: {all_combinations}")

    return model_name, accuracy, precision, recall, f1, all_combinations

# 데이터 필터링
first_half_2020 = data[(data['month'] >= '2019-01') & (data['month'] <= '2020-06')]
second_half_2020 = data[(data['month'] >= '2020-07') & (data['month'] <= '2020-12')]
first_half_2021 = data[(data['month'] >= '2021-01') & (data['month'] <= '2021-06')]
second_half_2021 = data[(data['month'] >= '2021-07') & (data['month'] <= '2021-12')]

# 시나리오별 데이터 준비 및 모델 학습/평가
scenarios = [
    ("Scenario 1 - First Half 2021", first_half_2020, second_half_2020, first_half_2021),
    ("Scenario 2 - Second Half 2021", first_half_2020, second_half_2020, second_half_2021),
    ("Scenario 3 - Second Half 2021", pd.concat([first_half_2020, second_half_2020], axis=0), first_half_2021, second_half_2021)
]

results = []

for scenario, train_data, test_data, eval_data in scenarios:
    X_train = prepare_data(train_data)
    X_test = prepare_data(test_data)
    X_eval = prepare_data(eval_data)

    X_train, _ = remove_no_mutations(X_train, X_train)
    X_test, _ = remove_no_mutations(X_test, X_test)
    X_eval, _ = remove_no_mutations(X_eval, X_eval)

    model_path = f"models/LightGBM_wave_{scenario.replace(' ', '_').replace('-', '').lower()}"
    train_lgb_model(X_train, X_train, model_path)
    results.append(evaluate_model(X_test, X_eval, model_path, f"LightGBM {scenario}"))

# 결과를 데이터프레임으로 정리
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "All Combinations"])

results_df.to_csv('LightGBM_wave_performance.txt', sep='\t', index=False)

# 모든 조합을 저장
all_combinations_df = pd.concat([
    pd.DataFrame({'Model': model_name, 'Combination': [combo], 'Frequency': [freq]})
    for model_name, _, _, _, _, combinations in results
    for combo, freq in combinations
], ignore_index=True)

all_combinations_df = all_combinations_df.sort_values(by=['Model', 'Frequency'], ascending=[True, False])

# 모든 조합을 파일로 저장
all_combinations_df.to_csv('wave_all_combinations_LightGBM.txt', sep='\t', index=False)
print("All combinations saved to wave_all_combinations_LightGBM.txt")
