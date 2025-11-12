#!/bin/bash
# 전체 파이프라인 자동 실행 스크립트
# 사용법: bash run_all.sh

set -e  # 에러 발생시 중단

echo "🚀 Dacon 프로젝트 전체 실행 시작"
echo "예상 소요 시간: 2-3시간"
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Phase 1: 환경 설정
echo "${YELLOW}=== Phase 1: 환경 설정 ===${NC}"
pip install -q ipykernel imbalanced-learn shap 2>/dev/null || echo "일부 패키지 이미 설치됨"
mkdir -p output/figures output/model_preds output/logs models/saved
echo "${GREEN}✅ 환경 설정 완료${NC}"
echo ""

# Phase 2: 데이터 검증
echo "${YELLOW}=== Phase 2: 데이터 검증 ===${NC}"
python -c "
import pandas as pd
train = pd.read_csv('data/train.csv')
print(f'데이터 크기: {train.shape}')
print(f'위험군 비율: {train[\"Label\"].mean():.2%}')
" > output/logs/data_check.txt
cat output/logs/data_check.txt
echo "${GREEN}✅ 데이터 검증 완료${NC}"
echo ""

# Phase 3: Baseline 모델 학습
echo "${YELLOW}=== Phase 3: Baseline 모델 학습 (30분 예상) ===${NC}"
python << 'SCRIPT'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
from catboost import CatBoostClassifier
import sys
sys.path.append('features')
sys.path.append('utils')
from feature_engineering import build_a_features, build_b_features
from preprocessing import FeaturePreprocessor

print("데이터 로드 및 Feature 생성 중...")
train = pd.read_csv('data/train.csv')
train_a = pd.read_csv('data/train/A.csv')
train_b = pd.read_csv('data/train/B.csv')

a_features = build_a_features(train_a)
b_features = build_b_features(train_b)

train_features = pd.concat([
    train[train['Test'] == 'A'].merge(a_features, on='Test_id', how='left'),
    train[train['Test'] == 'B'].merge(b_features, on='Test_id', how='left')
], ignore_index=True)

X = train_features.drop(['Test_id', 'Test', 'Label'], axis=1, errors='ignore')
y = train_features['Label']
# 숫자형 컬럼만 선택
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

# Preprocessor 초기화 및 학습
print("Preprocessor 학습 중...")
preprocessor = FeaturePreprocessor()
X = preprocessor.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Valid: {X_val.shape}")

# CatBoost Baseline
print("\nCatBoost 학습 중...")
cb = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=False
)
cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)

y_pred = cb.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred)
recall = recall_score(y_val, (y_pred > 0.5).astype(int))
f1 = f1_score(y_val, (y_pred > 0.5).astype(int))

print(f"\n✅ Baseline 완료")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

pd.DataFrame([{
    'model': 'CatBoost',
    'strategy': 'baseline',
    'roc_auc': roc_auc,
    'recall': recall,
    'f1': f1
}]).to_csv('output/baseline_results.csv', index=False)

import pickle
with open('models/saved/catboost_baseline.pkl', 'wb') as f:
    pickle.dump(cb, f)
with open('output/models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("✅ 모델 및 Preprocessor 저장 완료")
SCRIPT
echo "${GREEN}✅ Baseline 모델 학습 완료${NC}"
echo ""

# Phase 4: 불균형 처리
echo "${YELLOW}=== Phase 4: 불균형 처리 (SMOTE) ===${NC}"
python << 'SCRIPT'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import sys
sys.path.append('features')
from feature_engineering import build_a_features, build_b_features

print("데이터 준비 중...")
train = pd.read_csv('data/train.csv')
train_a = pd.read_csv('data/train/A.csv')
train_b = pd.read_csv('data/train/B.csv')

a_features = build_a_features(train_a)
b_features = build_b_features(train_b)

train_features = pd.concat([
    train[train['Test'] == 'A'].merge(a_features, on='Test_id', how='left'),
    train[train['Test'] == 'B'].merge(b_features, on='Test_id', how='left')
], ignore_index=True)

X = train_features.drop(['Test_id', 'Test', 'Label'], axis=1, errors='ignore')
y = train_features['Label']
# 숫자형 컬럼만 선택
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE 적용
print("SMOTE 적용 중...")
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"SMOTE 후 클래스 분포: {y_train_sm.mean():.3f}")

# 모델 학습
cb_smote = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=False
)
cb_smote.fit(X_train_sm, y_train_sm, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)

y_pred = cb_smote.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred)
recall = recall_score(y_val, (y_pred > 0.5).astype(int))
f1 = f1_score(y_val, (y_pred > 0.5).astype(int))

print(f"\n✅ SMOTE 완료")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

pd.DataFrame([{
    'strategy': 'SMOTE_0.3',
    'roc_auc': roc_auc,
    'recall': recall,
    'f1': f1
}]).to_csv('output/imbalance_results.csv', index=False)

import pickle
with open('models/saved/best_model.pkl', 'wb') as f:
    pickle.dump(cb_smote, f)
print("✅ 최적 모델 저장 완료")
SCRIPT
echo "${GREEN}✅ 불균형 처리 완료${NC}"
echo ""

# Phase 5: 제출 파일 생성
echo "${YELLOW}=== Phase 5: 제출 파일 생성 ===${NC}"
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
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())

# Preprocessor 로드 및 적용
print("Preprocessor 적용 중...")
with open('output/models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
X_test = preprocessor.transform(X_test)

print("모델 로드 및 예측 중...")
with open('models/saved/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'Test_id': test_features['Test_id'],
    'Label': y_pred
})
submission = submission.sort_values('Test_id').reset_index(drop=True)
submission.to_csv('output/submission.csv', index=False)

print(f"\n✅ 제출 파일 생성 완료")
print(f"Shape: {submission.shape}")
print(f"예측 평균: {y_pred.mean():.4f}")
SCRIPT
echo "${GREEN}✅ 제출 파일 생성 완료${NC}"
echo ""

# 최종 검증
echo "${YELLOW}=== 최종 검증 ===${NC}"
python -c "
import pandas as pd
sub = pd.read_csv('output/submission.csv')
test = pd.read_csv('data/test.csv')
print(f'제출 파일 shape: {sub.shape}')
print(f'테스트 파일 shape: {test.shape}')
print(f'Shape 일치: {sub.shape[0] == test.shape[0]}')
print(f'결측치: {sub.isna().sum().sum()}')
print(f'Label 범위: [{sub[\"Label\"].min():.4f}, {sub[\"Label\"].max():.4f}]')
"
echo ""

echo "${GREEN}🎉 모든 작업 완료!${NC}"
echo ""
echo "생성된 파일:"
echo "  - output/submission.csv (Dacon 제출용)"
echo "  - output/baseline_results.csv (Baseline 결과)"
echo "  - output/imbalance_results.csv (불균형 처리 결과)"
echo "  - models/saved/best_model.pkl (최적 모델)"
echo ""
echo "다음 단계:"
echo "  1. output/submission.csv를 Dacon에 제출하세요"
echo "  2. 리더보드 점수를 확인하세요"
echo "  3. 필요시 모델 개선을 진행하세요"
echo ""
echo "자세한 가이드: EXECUTION_GUIDE.md 참고"
