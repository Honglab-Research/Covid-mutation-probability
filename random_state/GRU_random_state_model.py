import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

# GPU 설정
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 데이터 로드 및 전처리
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
X, y = X[y != -1], tf.keras.utils.to_categorical(y[y != -1], num_classes=72)

# 데이터 분할
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# GRU 모델 학습 및 저장
def train_gru_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        GRU(64, input_shape=(X_train.shape[1], 1)),
        Dense(X_train.shape[1], activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        np.expand_dims(X_train, axis=-1), y_train,
        epochs=30, batch_size=2**16,
        validation_data=(np.expand_dims(X_val, axis=-1), y_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )

    model_path = 'models/GRU_random_state_model.h5'
    model.save(model_path)
    print(f"GRU model saved to {model_path}")

train_gru_model(X_train, y_train, X_val, y_val)

