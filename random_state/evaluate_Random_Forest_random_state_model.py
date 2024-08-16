import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 데이터 로드 및 전처리
data = pd.concat(pd.read_csv('filtered_data.txt', sep='\t', chunksize=100000), axis=0)
data = data[data['Nextstrain_clade'].str.strip() != 'recombinant'].drop(columns=['deletions', 'insertions'])

# Clade 계층 구조 및 변이 인코딩 함수
clade_hierarchy = {
    '19B': '19A', '20A': '19A', '20B': '20A', '21A': '20A', '20C': '20A', '21H': '20A', '21D': '20A',
    '21B': '20A', '20E': '20A', '21I': '21A', '21J': '21A', '21F': '20C', '21C': '20C', '20H': '20C',
    '20G': '20C', '21E': '20B', '20J': '20B', '20I': '20B', '20F': '20B', '20D': '20B', '21G': '20D',
    '21M': '20B', '21K': '21M', '21L': '21M', '23I': '21L', '22D': '21L', '23C': '22D', '22C': '21L',
    '22A': '21L', '22B': '21L', '22E': '22B', '22F': '21L', '23A': '22F', '23G': '23A', '23B': '22F',
    '23E': '22F', '23D': '22F', '23F': '23D', '23H': '23F'
}

def encode_mutations(mutations, start=437, end=508):
    encoded = np.zeros(end - start + 1, dtype=np.int8)
    for mut in str(mutations).split(','):
        if mut and start <= int(mut[1:-1]) <= end:
            encoded[int(mut[1:-1]) - start] = 1
    return encoded

# 데이터 필터링 및 인코딩
filtered_data = data[data['Nextstrain_clade'].isin(clade_hierarchy.values())]
y_true = filtered_data['Nextstrain_clade'].map(clade_hierarchy)
X_filtered = np.array(filtered_data['aaSubstitutions'].apply(lambda x: encode_mutations(x, 437, 508)).tolist())
y_true_labels = np.argmax(pd.get_dummies(y_true).values, axis=1)

# Random Forest 모델 로드 및 성능 평가
model = joblib.load('models/Random_Forest_random_state_model.pkl')
y_pred_labels = model.predict(X_filtered)

# 성능 평가 및 저장
metrics = {
    'Accuracy': accuracy_score(y_true_labels, y_pred_labels),
    'Precision': precision_score(y_true_labels, y_pred_labels, average='weighted'),
    'Recall': recall_score(y_true_labels, y_pred_labels, average='weighted'),
    'F1 Score': f1_score(y_true_labels, y_pred_labels, average='weighted')
}

with open('Random_Forest_random_state_performance.txt', 'w') as f:
    for metric, value in metrics.items():
        f.write(f"Random Forest Model {metric}: {value:.4f}\n")

print("Performance metrics have been saved to Random_Forest_random_state_performance.txt.")

