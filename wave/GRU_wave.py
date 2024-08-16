import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# GPU 설정
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# 변이 위치가 없는 경우 제거
def remove_no_mutations(X):
    return X[np.any(X != 0, axis=1)]

# GRU 모델 학습
def train_gru_model(X_train, y_train, model_path):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(64, input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.layers.Dense(X_train.shape[1], activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=2**16, validation_split=0.2, callbacks=[early_stopping])

    model.save(model_path)
    print(f"GRU model saved to {model_path}")

# 모델 성능 평가 및 자주 나온 조합 출력
def evaluate_model(X_input, X_target, model_path, model_name):
    model = tf.keras.models.load_model(model_path)
    X_input_gru = np.expand_dims(X_input, axis=-1)
    y_pred = model.predict(X_input_gru)
    y_pred_labels = (y_pred > 0.5).astype(int)

    if len(y_pred_labels) > len(X_target):
        y_pred_labels = y_pred_labels[:len(X_target)]
    else:
        X_target = X_target[:len(y_pred_labels)]

    accuracy = accuracy_score(X_target, y_pred_labels)
    precision = precision_score(X_target, y_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(X_target, y_pred_labels, average='weighted')
    f1 = f1_score(X_target, y_pred_labels, average='weighted')

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1 Score: {f1:.4f}")

    total_predictions = len(y_pred_labels)
    pred_counter = Counter([tuple(np.where(row == 1)[0] + 437) for row in y_pred_labels])
    top_5_combinations = pred_counter.most_common(5)
    top_5_combinations = [(combo, count / total_predictions) for combo, count in top_5_combinations]
    print(f"{model_name} Top 5 Mutations Combinations: {top_5_combinations}")

    return model_name, accuracy, precision, recall, f1, top_5_combinations

# 데이터 필터링
first_half_2020 = data[(data['month'] >= '2019-01') & (data['month'] <= '2020-06')]
second_half_2020 = data[(data['month'] >= '2020-07') & (data['month'] <= '2020-12')]
first_half_2021 = data[(data['month'] >= '2021-01') & (data['month'] <= '2021-06')]
second_half_2021 = data[(data['month'] >= '2021-07') & (data['month'] <= '2021-12')]

# 시나리오 1: 2020년 상반기 데이터로 2020년 하반기 예측 후, 2021년 상반기 평가
train_data_1 = first_half_2020
test_data_1 = second_half_2020
eval_data_1 = first_half_2021

X_train_1 = prepare_data(train_data_1)
X_test_1 = prepare_data(test_data_1)
X_eval_1 = prepare_data(eval_data_1)

X_train_1 = remove_no_mutations(X_train_1)
X_test_1 = remove_no_mutations(X_test_1)
X_eval_1 = remove_no_mutations(X_eval_1)

# GRU 모델 학습 및 평가 for scenario 1
model_path_1 = 'models/gru_model_scenario_1.h5'
train_gru_model(np.expand_dims(X_train_1, axis=-1), X_train_1, model_path_1)

# 결과를 저장할 리스트 초기화
results = []

print("Evaluating GRU model for first_half_2021")
results.append(evaluate_model(X_test_1, X_eval_1, model_path_1, "GRU Scenario 1 - First Half 2021"))

# 시나리오 2: 2020년 상반기 데이터로 2020년 하반기 예측 후, 2021년 하반기 평가
eval_data_2 = second_half_2021

X_eval_2 = prepare_data(eval_data_2)
X_eval_2 = remove_no_mutations(X_eval_2)

print("Evaluating GRU model for second_half_2021")
results.append(evaluate_model(X_test_1, X_eval_2, model_path_1, "GRU Scenario 2 - Second Half 2021"))

# 시나리오 3: 2020년 전체 데이터로 2021년 상반기 예측 후, 2021년 하반기 평가
train_data_3 = pd.concat([first_half_2020, second_half_2020], axis=0)
test_data_3 = first_half_2021

X_train_3 = prepare_data(train_data_3)
X_test_3 = prepare_data(test_data_3)

X_train_3 = remove_no_mutations(X_train_3)
X_test_3 = remove_no_mutations(X_test_3)

# GRU 모델 학습 및 평가 for scenario 3
model_path_3 = 'models/gru_model_scenario_3.h5'
train_gru_model(np.expand_dims(X_train_3, axis=-1), X_train_3, model_path_3)

print("Evaluating GRU model for second_half_2021 with extended training data")
results.append(evaluate_model(X_test_3, X_eval_2, model_path_3, "GRU Scenario 3 - Second Half 2021"))

# 결과를 데이터프레임으로 정리
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Top 5 Combinations"])

results_df.to_csv('GRU_wave_performance.txt', sep='\t', index=False)
