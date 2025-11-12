#!/bin/bash
# 교차검증으로 저장된 모델들(cv_models.pkl)을 이용해 테스트 세트 앙상블 예측
# 평균, 가중 평균(각 fold final F1 기준) 모두 산출
# 사용: bash run_cv_predict.sh

set -e

echo "🚀 CV 앙상블 예측 시작"

python <<'PY'
import pickle, os, numpy as np, pandas as pd
import sys
sys.path.append('utils'); sys.path.append('features')
from common_utils import load_data, save_submission
from feature_engineering import build_a_features, build_b_features, merge_features_with_labels
from preprocessing import FeaturePreprocessor

# 로드 확인
models_path = 'output/models/cv_models.pkl'
preproc_path = 'output/models/cv_preprocessor.pkl'
thr_path = 'output/models/cv_final_threshold.pkl'
fold_metrics_path = 'output/cv_fold_metrics.csv'

if not os.path.exists(models_path):
    raise FileNotFoundError('cv_models.pkl 없음: 먼저 run_cv_pipeline.sh 실행 필요')
if not os.path.exists(preproc_path):
    raise FileNotFoundError('cv_preprocessor.pkl 없음: 먼저 run_cv_pipeline.sh 실행 필요')

print('[1] 테스트 데이터 로드 & 피처 생성')
train, test, train_a, train_b, test_a, test_b = load_data()
fa_test = build_a_features(test_a)
fb_test = build_b_features(test_b)

test_a_merged = test[test['Test']=='A'].merge(fa_test,on='Test_id',how='left')
test_b_merged = test[test['Test']=='B'].merge(fb_test,on='Test_id',how='left')
X_test_full = pd.concat([test_a_merged, test_b_merged], ignore_index=True)
ids = X_test_full['Test_id'].values
X_test = X_test_full.drop(['Test_id','Test'], axis=1, errors='ignore')

print('[2] 전처리기 로드')
pre = FeaturePreprocessor.load(preproc_path)
X_test = pre.transform(X_test)
print('전처리 후 테스트 Shape:', X_test.shape)

print('[3] 모델 로드')
with open(models_path,'rb') as f:
    models = pickle.load(f)
print(f'Model count: {len(models)}')

print('[4] Fold별 확률 예측')
fold_preds = []
for i,m in enumerate(models,1):
    p = m.predict_proba(X_test)[:,1]
    fold_preds.append(p)
    print(f'  Fold {i} 예측 범위: [{p.min():.4f},{p.max():.4f}] mean={p.mean():.4f}')

pred_matrix = np.vstack(fold_preds)  # shape (K, N)
mean_pred = pred_matrix.mean(axis=0)

print('[5] 가중 평균 (F1 기반) 계산')
if os.path.exists(fold_metrics_path):
    fm = pd.read_csv(fold_metrics_path)
    # F1이 0이면 최소 가중치 보호
    raw_weights = fm['val_f1_final'].values
    safe_weights = np.where(raw_weights<=0, 1e-6, raw_weights)
    weights = safe_weights / safe_weights.sum()
    weighted_pred = (pred_matrix.T * weights).sum(axis=1)
    print('Fold weights:', weights)
else:
    print('cv_fold_metrics.csv 없음: 가중 평균 대신 단순 평균만 사용')
    weighted_pred = mean_pred

# 최종 저장: 평균/가중 평균 둘 다
sub_mean = save_submission(ids, mean_pred, 'output/submissions/submission_cv_mean.csv')
sub_weighted = save_submission(ids, weighted_pred, 'output/submissions/submission_cv_weighted.csv')
print('\n✅ 앙상블 제출 파일 생성 완료')
print(' mean.csv  범위: [{:.4f},{:.4f}] mean={:.4f}'.format(mean_pred.min(), mean_pred.max(), mean_pred.mean()))
print(' weighted.csv 범위: [{:.4f},{:.4f}] mean={:.4f}'.format(weighted_pred.min(), weighted_pred.max(), weighted_pred.mean()))
PY

echo "🎉 완료: output/submissions/submission_cv_mean.csv / submission_cv_weighted.csv"
