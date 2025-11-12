#!/bin/bash
# 스태킹 모델(CatBoost + LightGBM + XGBoost + 메타 로지스틱)로 테스트 세트 예측
# 사용: bash run_stacking_predict.sh

set -e

echo "🚀 스태킹 테스트 예측 시작"

python <<'PY'
import pickle, os, numpy as np, pandas as pd
import sys
sys.path.append('utils'); sys.path.append('features')
from common_utils import load_data, save_submission
from feature_engineering import build_a_features, build_b_features
from preprocessing import FeaturePreprocessor

# 저장된 모델/전처리기 확인
required_files = [
    'output/models/stack_preprocessor.pkl',
    'output/models/stack_cat_models.pkl',
    'output/models/stack_lgb_models.pkl',
    'output/models/stack_xgb_models.pkl',
    'output/models/stack_meta_model.pkl'
]
for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f'{f} 없음. 먼저 run_stacking_pipeline.sh 실행 필요')

print('[1] 테스트 데이터 로드 & 피처 생성')
train, test, train_a, train_b, test_a, test_b = load_data()
fa_test = build_a_features(test_a)
fb_test = build_b_features(test_b)

test_a_merged = test[test['Test']=='A'].merge(fa_test, on='Test_id', how='left')
test_b_merged = test[test['Test']=='B'].merge(fb_test, on='Test_id', how='left')
X_test_full = pd.concat([test_a_merged, test_b_merged], ignore_index=True)
test_ids = X_test_full['Test_id'].values
X_test_raw = X_test_full.drop(['Test_id','Test'], axis=1, errors='ignore')

print('[2] 전처리기 로드 및 적용')
pre = FeaturePreprocessor.load('output/models/stack_preprocessor.pkl')
X_test = pre.transform(X_test_raw)
print(f'전처리 후 Shape: {X_test.shape}')

print('[3] Base 모델 로드')
with open('output/models/stack_cat_models.pkl','rb') as f:
    cat_models = pickle.load(f)
with open('output/models/stack_lgb_models.pkl','rb') as f:
    lgb_models = pickle.load(f)
with open('output/models/stack_xgb_models.pkl','rb') as f:
    xgb_models = pickle.load(f)
print(f'Loaded: {len(cat_models)} CatBoost, {len(lgb_models)} LightGBM, {len(xgb_models)} XGBoost')

print('[4] Base 모델 예측 (Fold 평균)')
cat_preds = np.vstack([m.predict_proba(X_test)[:,1] for m in cat_models]).mean(axis=0)
lgb_preds = np.vstack([m.predict_proba(X_test)[:,1] for m in lgb_models]).mean(axis=0)
xgb_preds = np.vstack([m.predict_proba(X_test)[:,1] for m in xgb_models]).mean(axis=0)

print(f'  CatBoost 범위: [{cat_preds.min():.4f}, {cat_preds.max():.4f}]')
print(f'  LightGBM 범위: [{lgb_preds.min():.4f}, {lgb_preds.max():.4f}]')
print(f'  XGBoost  범위: [{xgb_preds.min():.4f}, {xgb_preds.max():.4f}]')

print('[5] 메타 모델 로드 및 최종 예측')
with open('output/models/stack_meta_model.pkl','rb') as f:
    meta_model = pickle.load(f)

meta_X_test = pd.DataFrame({'cat': cat_preds, 'lgb': lgb_preds, 'xgb': xgb_preds})
final_preds = meta_model.predict_proba(meta_X_test)[:,1]

print(f'  Stacking 범위: [{final_preds.min():.4f}, {final_preds.max():.4f}]')
print(f'  Stacking 평균: {final_preds.mean():.4f}')

print('[6] 제출 파일 저장')
submission = save_submission(test_ids, final_preds, 'output/submissions/submission_stacking.csv')

print('\n✅ 스태킹 제출 파일 생성 완료!')
print('output/submissions/submission_stacking.csv')
PY

echo ""
echo "🎉 완료! 다음 제출 파일 중 선택하세요:"
echo "  1. submission.csv (Baseline 단일)"
echo "  2. submission_cv_mean.csv (CV 평균)"
echo "  3. submission_cv_weighted.csv (CV 가중)"
echo "  4. submission_stacking.csv (멀티모델 스태킹) ← 최신"
