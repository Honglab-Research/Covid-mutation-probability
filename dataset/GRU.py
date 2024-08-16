import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from collections import Counter

# GPU 설정
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# GRU 모델 학습 실행
def train_all_gru_models(data, model_paths):
    for idx, (dataset_name, (X, y)) in enumerate(data.items(), start=1):
        X = remove_no_mutations(X)
        model_path = model_paths[f'Training dataset{idx}']
        X_train_gru = np.expand_dims(X, axis=-1)
        train_gru_model(X_train_gru, X, model_path)

# GRU 모델 예측 및 결과 저장
def predict_and_save_results(data, model_paths):
    clades_24AB = ['24A', '24B']
    X_24AB = prepare_data(data, clades_24AB)[0]
    X_24AB_gru = np.expand_dims(X_24AB, axis=-1)

    results = []
    for idx, (model_name, model_path) in enumerate(model_paths.items(), start=1):
        predictions_24AB = predict_mutations(X_24AB_gru, model_path)
        all_combinations_24AB = get_all_combinations(predictions_24AB, len(X_24AB))

        for combo, freq in all_combinations_24AB:
            results.append({
                'Model': f'Training dataset{idx}',
                'Clade': '24A/24B',
                'Combination': combo,
                'Frequency': freq
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv('GRU_all_combinations_24AB.csv', index=False)
    print(results_df)

# 모든 조합 및 빈도 계산 함수
def get_all_combinations(predictions, total_samples):
    pred_combinations = [''.join(map(str, np.round(pred).astype(int))) for pred in predictions]
    combination_counts = Counter(pred_combinations)
    all_combinations = combination_counts.most_common()
    all_combinations = [(convert_to_positions(combo), count / total_samples) for combo, count in all_combinations]
    return all_combinations

# 조합을 변이 위치로 변환 함수
def convert_to_positions(combo):
    positions = []
    for i, bit in enumerate(combo):
        if bit == '1':
            positions.append(str(i + 437))
    return ', '.join(positions)

# 모델 로드 및 예측
def predict_mutations(X_test, model_path):
    model = load_model(model_path)
    predictions = model.predict(X_test)
    return predictions

# 통합 실행
file_path = 'filtered_data.txt'
data = load_and_preprocess_data(file_path)

# 모델 경로 설정
model_paths = {
    'Training dataset1': 'models/Training_dataset1_GRU.h5',
    'Training dataset2': 'models/Training_dataset2_GRU.h5',
    'Training dataset3': 'models/Training_dataset3_GRU.h5',
    'Training dataset4': 'models/Training_dataset4_GRU.h5',
    'Training dataset5': 'models/Training_dataset5_GRU.h5',
    'Training dataset6': 'models/Training_dataset6_GRU.h5'
}

# 데이터셋 준비
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
    'before_omicron': prepare_data(data, clades_before_omicron),
    'till_ba1_ba2': prepare_data(data, clades_till_ba1_ba2),
    'till_ba4_ba5': prepare_data(data, clades_till_ba4_ba5),
    'till_xbb_bq': prepare_data(data, clades_till_xbb_bq),
    'omicron_BA.1_2': prepare_data(data, omicron1),
    'omicron-BA.1_2_4_5': prepare_data(data, omicron2)
}

# GRU 모델 학습
train_all_gru_models(data_sets, model_paths)

# 예측 수행 및 결과 저장
predict_and_save_results(data, model_paths)

