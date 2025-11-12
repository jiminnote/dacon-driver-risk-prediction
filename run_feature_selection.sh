#!/bin/bash
# CatBoost 교차검증 모델 중요도 집계 후 Top-N 피처 재학습
# 사용: bash run_feature_selection.sh [TOP_N]
# 기본 TOP_N=150

set -e
TOP_N=${1:-150}

echo "🚀 Feature Importance 집계 및 Top-${TOP_N} 재학습 시작"

python <<PY
import os, pickle, numpy as np, pandas as pd
import sys
sys.path.append('utils'); sys.path.append('features')
from common_utils import load_data
from feature_engineering import build_a_features, build_b_features, merge_features_with_labels
from preprocessing import FeaturePreprocessor
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# 모델/전처리기 로드
models_path = 'output/models/cv_models.pkl'
preproc_path = 'output/models/cv_preprocessor.pkl'
if not os.path.exists(models_path):
    raise FileNotFoundError('cv_models.pkl 없음. 먼저 run_cv_pipeline.sh 실행')
if not os.path.exists(preproc_path):
    raise FileNotFoundError('cv_preprocessor.pkl 없음. 먼저 run_cv_pipeline.sh 실행')

with open(models_path,'rb') as f:
    models = pickle.load(f)
from preprocessing import FeaturePreprocessor
pre = FeaturePreprocessor.load(preproc_path)

# 데이터 로드 및 전체 피처 생성 (preprocessor 기준 사용안함, 중요도용 원본)
train, test, train_a, train_b, test_a, test_b = load_data()
fa = build_a_features(train_a)
fb = build_b_features(train_b)
X_raw, y_full = merge_features_with_labels(train, fa, fb)
X_full = pre.transform(X_raw)

# 중요도 집계 (각 모델 CatBoost 중요도 길이가 동일하다고 가정)
importances = []
for i,m in enumerate(models,1):
    imp = m.get_feature_importance(type='FeatureImportance')
    if len(imp) != X_full.shape[1]:
        print(f'⚠️ Fold {i} 중요도 길이 불일치: {len(imp)} vs {X_full.shape[1]}')
    importances.append(imp)

imp_matrix = np.vstack(importances)
mean_imp = imp_matrix.mean(axis=0)
feat_df = pd.DataFrame({'feature': X_full.columns, 'importance_mean': mean_imp})
feat_df = feat_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
feat_df.to_csv('output/feature_importance_mean.csv', index=False)

selected = feat_df.head(int(${TOP_N}))['feature'].tolist()
print(f'Top-{${TOP_N}} 피처 선택 완료. 예시 상위 5개:', selected[:5])

# 선택 피처로 재학습 (단일 CatBoost, 전처리 데이터 기준)
X_sel = X_full[selected]
X_train, X_val, y_train, y_val = train_test_split(X_sel, y_full, test_size=0.2, random_state=42, stratify=y_full)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.04,
    depth=6,
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=100
)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

proba_val = model.predict_proba(X_val)[:,1]
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_val, proba_val)

# 간단 임계값 (0.5) 평가
bin_val = (proba_val >= 0.5).astype(int)
prec = precision_score(y_val, bin_val, zero_division=0)
rec = recall_score(y_val, bin_val, zero_division=0)
f1 = f1_score(y_val, bin_val, zero_division=0)

print(f'Retrain AUC={auc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}')

with open('output/models/feature_selected_model.pkl','wb') as f:
    pickle.dump(model,f)

pd.DataFrame([{'auc':auc,'precision':prec,'recall':rec,'f1':f1,'top_n':${TOP_N}}]).to_csv('output/feature_selected_results.csv',index=False)
print('✅ 저장 완료: feature_selected_model.pkl, feature_selected_results.csv, feature_importance_mean.csv')
PY

echo "🎉 Feature Selection 재학습 완료"
