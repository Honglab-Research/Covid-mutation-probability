import pandas as pd
import numpy as np
import re

file_path = 'data.txt'

data = pd.read_csv(file_path, sep='\t')

def process_aa_mutations(AA_mutation):
    if pd.isna(AA_mutation):
        return np.nan
    mutations = [m.replace('S:', '') for m in str(AA_mutation).split(',') if m.startswith('S:')]
    mutations = [m for m in mutations if 437 <= int(m[1:-1]) <= 509]
    if not mutations or "*" in ''.join(mutations):
        return np.nan
    return ','.join(mutations)

def clean_clade(clade):
    return str(clade).split('(')[0] if '(' in str(clade) else str(clade)

def clean_lineage(lineage):
    return np.nan if lineage in ['', '?'] else lineage

def is_valid_date(date_str):
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', str(date_str)))

# 데이터 전처리
data['Nextstrain_clade'] = data['Nextstrain_clade'].apply(clean_clade)
data['aaSubstitutions'] = data['aaSubstitutions'].apply(process_aa_mutations)

# 유효한 aaSubstitutions과 날짜만 남기기
data = data.dropna(subset=['aaSubstitutions'])
data = data[data['date'].apply(is_valid_date)]

# 날짜 처리 및 필터링
reference_date = pd.to_datetime("2019-12-22")
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data['days'] = (data['date'] - reference_date).dt.days
data = data.sort_values(by='days', ascending=True)

# 인간 호스트 데이터 필터링
use_host = ['Homo sapiens', 'Homo']
human_data = data[data['host'].isin(use_host)]

# Pango_lineage 정리 및 불필요한 열 제거
human_data['pango_lineage'] = human_data['pango_lineage'].apply(clean_lineage)
human_data = human_data.dropna(subset=['pango_lineage', 'Nextstrain_clade'])

# 필요한 열만 선택하여 저장
columns_to_keep = ['date', 'region', 'Nextstrain_clade', 'aaSubstitutions', 'days']
human_data = human_data[columns_to_keep]

human_data.to_csv('filtered_data.txt.txt', sep='\t', mode='w', index=False)

