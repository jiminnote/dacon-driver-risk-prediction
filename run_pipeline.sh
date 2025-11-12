#!/bin/bash
# 전체 파이프라인 자동 실행 스크립트
# 사용법: bash run_pipeline.sh

set -e  # 에러 발생시 중단



# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 환경 설정
echo "${YELLOW}=== 환경 설정 ===${NC}"
pip install -q pandas numpy matplotlib seaborn scikit-learn catboost xgboost lightgbm imbalanced-learn 2>/dev/null || true
mkdir -p ./output/figures ./output/models ./output/submissions
echo "${GREEN} 환경 설정 완료${NC}"
echo ""

# Baseline 모델 학습
echo "${YELLOW}=== Baseline 모델 학습 (예상 30-40분, 진행 로그 100iter 간격 출력) ===${NC}"
python << 'SCRIPT'
import sys
sys.path.append('utils')
sys.path.append('features')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from catboost import CatBoostClassifier
import pickle
import os, time

from common_utils import load_data, evaluate_model
from feature_engineering import build_a_features, build_b_features, merge_features_with_labels
from preprocessing import FeaturePreprocessor

print("데이터 로드 및 Feature Engineering...")
train, test, train_a, train_b, test_a, test_b = load_data()
features_a = build_a_features(train_a)
features_b = build_b_features(train_b)
X, y = merge_features_with_labels(train, features_a, features_b)

# Train 기준 전처리기 적합
preprocessor = FeaturePreprocessor().fit(X)
X = preprocessor.transform(X)

print(f"\n최종 학습 데이터: X {X.shape}, y {y.shape}")
print(f"피처 개수: {X.shape[1]}개")
print(f"위험군 비율: {y.mean():.2%}")

# Train/Valid Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CatBoost with Auto Class Weights (불균형 처리)
print("\nCatBoost 학습 중 (Auto Class Weights)...")
use_gpu = os.environ.get('CATBOOST_GPU','0') == '1'
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    auto_class_weights='Balanced',  # 불균형 자동 조정
    random_seed=42,
    task_type='GPU' if use_gpu else 'CPU',
    verbose=100
)
start_train = time.time()
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
train_time = time.time() - start_train
print(f"학습 소요 시간: {train_time/60:.1f}분")
try:
    print(f"Best iteration: {model.get_best_iteration()}")
except Exception:
    pass

# 평가
y_pred_proba = model.predict_proba(X_val)[:, 1]

# 최적 임계값 탐색 (F1 기준)
print("\n임계값 다중 기준 탐색...")
thresholds = np.arange(0.01, 0.99, 0.01)
records = []
for t in thresholds:
    y_bin = (y_pred_proba >= t).astype(int)
    prec = precision_score(y_val, y_bin, zero_division=0)
    rec = recall_score(y_val, y_bin, zero_division=0)
    f1 = f1_score(y_val, y_bin, zero_division=0)
    # Youden's J = TPR - FPR
    tp = ((y_val==1) & (y_bin==1)).sum()
    fn = ((y_val==1) & (y_bin==0)).sum()
    fp = ((y_val==0) & (y_bin==1)).sum()
    tn = ((y_val==0) & (y_bin==0)).sum()
    tpr = tp / (tp + fn) if (tp+fn)>0 else 0
    fpr = fp / (fp + tn) if (fp+tn)>0 else 0
    youden = tpr - fpr
    records.append({'threshold': t, 'precision': prec, 'recall': rec, 'f1': f1, 'youden': youden})

thr_df = pd.DataFrame(records)

# 1) F1 최고
best_f1_row = thr_df.loc[thr_df['f1'].idxmax()]
best_f1_thr = float(best_f1_row.threshold)

# 2) Recall@Precision>=0.08 중 Recall 최고
candidate = thr_df[thr_df['precision'] >= 0.08]
if candidate.empty:
    best_rp_thr = best_f1_thr  # fallback
    best_rp_row = best_f1_row
else:
    best_rp_row = candidate.sort_values('recall', ascending=False).iloc[0]
    best_rp_thr = float(best_rp_row.threshold)

# 3) Youden J 최고
best_youden_row = thr_df.loc[thr_df['youden'].idxmax()]
best_youden_thr = float(best_youden_row.threshold)

print(f"F1 기준 최적: thr={best_f1_thr:.3f}, F1={best_f1_row.f1:.4f}, Prec={best_f1_row.precision:.4f}, Rec={best_f1_row.recall:.4f}")
print(f"Recall@Precision≥0.08 최적: thr={best_rp_thr:.3f}, Rec={best_rp_row.recall:.4f}, Prec={best_rp_row.precision:.4f}, F1={best_rp_row.f1:.4f}")
print(f"Youden J 최적: thr={best_youden_thr:.3f}, J={best_youden_row.youden:.4f}, Rec={best_youden_row.recall:.4f}, Prec={best_youden_row.precision:.4f}")

# 최종 선택 전략: 우선 Recall@Precision≥0.08, 없으면 F1, 그다음 Youden
final_thr = best_rp_thr if candidate.shape[0] > 0 else best_f1_thr if best_f1_row.f1 >= best_youden_row.f1 else best_youden_thr
print(f"선택된 최종 임계값: {final_thr:.3f}")

# 임계값 탐색 테이블 저장
thr_df.to_csv('output/thresholds_baseline.csv', index=False)

metrics = evaluate_model(y_val, y_pred_proba, threshold=final_thr, model_name='CatBoost Balanced')

# 저장
with open('output/models/baseline_best_model.pkl', 'wb') as f:
    pickle.dump(model, f)
preprocessor.save('output/models/preprocessor.pkl')
with open('output/models/best_threshold.pkl', 'wb') as f:
    pickle.dump(final_thr, f)

pd.DataFrame([metrics]).to_csv('output/baseline_results.csv', index=False)
print("\nBaseline 모델 학습 완료")
print(f"ROC AUC: {metrics['roc_auc']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
SCRIPT
echo "${GREEN}Baseline 완료${NC}"
echo ""

# 제출 파일 생성
echo "${YELLOW}=== 제출 파일 생성 ===${NC}"
python << 'SCRIPT'
import sys
sys.path.append('utils')
sys.path.append('features')

import pandas as pd
import numpy as np
import pickle

from common_utils import load_data, save_submission
from feature_engineering import build_a_features, build_b_features
from preprocessing import FeaturePreprocessor

print("테스트 데이터 로드...")
train, test, train_a, train_b, test_a, test_b = load_data()

print("Feature Engineering...")
test_features_a = build_a_features(test_a)
test_features_b = build_b_features(test_b)

test_a_merged = test[test['Test'] == 'A'].merge(test_features_a, on='Test_id', how='left')
test_b_merged = test[test['Test'] == 'B'].merge(test_features_b, on='Test_id', how='left')
test_merged = pd.concat([test_a_merged, test_b_merged], ignore_index=True)

test_ids = test_merged['Test_id'].values
X_test = test_merged.drop(['Test_id', 'Test'], axis=1, errors='ignore')

try:
    preprocessor = FeaturePreprocessor.load('output/models/preprocessor.pkl')
    print("저장된 전처리기 로드 완료")
except FileNotFoundError:
    print("⚠️ preprocessor.pkl이 없어 train 기반으로 즉시 생성합니다 (1회성).")
    # Train 데이터에서 전처리 기준 산출
    train_features_a = build_a_features(train_a)
    train_features_b = build_b_features(train_b)
    from feature_engineering import merge_features_with_labels
    X_train_all, _ = merge_features_with_labels(train, train_features_a, train_features_b)
    preprocessor = FeaturePreprocessor().fit(X_train_all)
    # 다음 실행을 위해 저장
    preprocessor.save('output/models/preprocessor.pkl')
    print("임시 전처리기 생성 및 저장 완료: output/models/preprocessor.pkl")

X_test = preprocessor.transform(X_test)

print(f"전처리 후 테스트 피처: {X_test.shape}")

print("모델 로드 및 예측...")
with open('output/models/baseline_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 최적 임계값 로드 (없으면 확률값 그대로 사용)
try:
    with open('output/models/best_threshold.pkl', 'rb') as f:
        best_threshold = pickle.load(f)
    print(f"최적 임계값 로드: {best_threshold:.3f}")
except:
    best_threshold = None
    print("임계값 없음 → 확률값으로 제출")

predictions = model.predict_proba(X_test)[:, 1]

# 대회 평가는 ROC AUC이므로 확률값 제출이 기본. 필요시 참조용으로 이진 예측도 산출 가능.
submission = save_submission(
    test_ids=test_ids,
    predictions=predictions,
    file_path='output/submissions/submission.csv'
)

print("\n✅ 제출 파일 생성 완료!")
print(f"예측 확률 범위: [{predictions.min():.4f}, {predictions.max():.4f}]")
print(f"예측 평균: {predictions.mean():.4f}")
SCRIPT
echo "${GREEN}✅ 제출 파일 생성 완료${NC}"
echo ""

# 최종 검증
echo "${YELLOW}=== 최종 검증 ===${NC}"
python << 'SCRIPT'
import pandas as pd

sub = pd.read_csv('output/submissions/submission.csv')
test = pd.read_csv('data/test.csv')

print("제출 파일 검증:")
print(f"  Shape: {sub.shape} (Test: {test.shape})")
print(f"  Shape 일치: {sub.shape[0] == test.shape[0]}")
print(f"  결측치: {sub.isnull().sum().sum()}")
print(f"  Label 범위: [{sub['Label'].min():.4f}, {sub['Label'].max():.4f}]")
print(f"  Label 평균: {sub['Label'].mean():.4f}")

if sub.shape[0] == test.shape[0] and sub.isnull().sum().sum() == 0:
    print("\n모든 검증 통과!")
else:
    print("\n검증 실패!")
SCRIPT
echo ""

echo "${GREEN 전체 파이프라인 실행 완료!${NC}"
echo ""
echo "생성된 파일:"
echo "  - output/submissions/submission.csv (Dacon 제출용)"
echo "  - output/models/baseline_best_model.pkl (학습된 모델)"
echo "  - output/baseline_results.csv (평가 결과)"
echo ""
echo "다음 단계:"
echo "  1. output/submissions/submission.csv를 Dacon에 제출하세요"
echo "  2. 추가 개선을 위해 modeling 폴더의 다른 노트북을 실행하세요"
