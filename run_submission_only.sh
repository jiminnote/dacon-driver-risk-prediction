#!/bin/bash

# 제출 파일 생성만 실행
echo "🚀 제출 파일 생성 시작"

python << 'SCRIPT'
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('features')
sys.path.append('utils')
from feature_engineering import build_a_features, build_b_features
from preprocessing import FeaturePreprocessor

print("테스트 데이터 로드 중...")
test = pd.read_csv('data/test.csv')
test_a = pd.read_csv('data/test/A.csv')
test_b = pd.read_csv('data/test/B.csv')

print("Feature 생성 중...")
a_features_test = build_a_features(test_a)
b_features_test = build_b_features(test_b)

test_features = pd.concat([
    test[test['Test'] == 'A'].merge(a_features_test, on='Test_id', how='left'),
    test[test['Test'] == 'B'].merge(b_features_test, on='Test_id', how='left')
], ignore_index=True)

X_test = test_features.drop(['Test_id', 'Test'], axis=1, errors='ignore')
# 숫자형 컬럼만 선택
X_test = X_test.select_dtypes(include=[np.number])
X_test = X_test.replace([np.inf, -np.inf], np.nan)

print(f"Test features shape before preprocessing: {X_test.shape}")

# Preprocessor 로드 및 적용
print("Preprocessor 적용 중...")
try:
    with open('output/models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    X_test = preprocessor.transform(X_test)
    print(f"Test features shape after preprocessing: {X_test.shape}")
except FileNotFoundError:
    print("⚠️ Preprocessor 파일 없음. median imputation만 수행...")
    X_test = X_test.fillna(X_test.median())

print("모델 로드 및 예측 중...")
with open('models/saved/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'Test_id': test_features['Test_id'],
    'Label': y_pred
})

submission.to_csv('output/submissions/submission.csv', index=False)
print(f"✅ 제출 파일 생성 완료: output/submissions/submission.csv")
print(f"예측값 통계 - Min: {y_pred.min():.4f}, Max: {y_pred.max():.4f}, Mean: {y_pred.mean():.4f}")
SCRIPT

echo "✅ 완료"
