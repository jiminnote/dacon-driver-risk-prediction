#!/bin/bash
# 교차검증 기반 CatBoost 학습 및 임계값 탐색 스크립트
# 사용: bash run_cv_pipeline.sh  (CATBOOST_GPU=1 환경변수 설정시 GPU 사용)

set -e

echo "🚀 교차검증 파이프라인 시작"

python <<'PY'
import os, pickle, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from catboost import CatBoostClassifier

import sys
sys.path.append('utils')
sys.path.append('features')

from common_utils import load_data, evaluate_model
from feature_engineering import build_a_features, build_b_features, merge_features_with_labels
from preprocessing import FeaturePreprocessor

print('[1] 데이터 로드 & 피처 생성')
train, test, train_a, train_b, test_a, test_b = load_data()
fa = build_a_features(train_a)
fb = build_b_features(train_b)
X_all, y_all = merge_features_with_labels(train, fa, fb)

print('[2] 전처리기 적합')
pre = FeaturePreprocessor().fit(X_all)
X_all = pre.transform(X_all)
print('전체 데이터 Shape:', X_all.shape)

print('[3] Stratified KFold 준비')
K=5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

use_gpu = os.environ.get('CATBOOST_GPU','0') == '1'

oof_preds = np.zeros(len(y_all))
fold_metrics = []
models = []

for fold,(tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
	print(f'\n▶ Fold {fold}/{K}')
	X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
	y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]

	model = CatBoostClassifier(
		iterations=400,
		learning_rate=0.05,
		depth=6,
		auto_class_weights='Balanced',
		random_seed=fold+100,
		task_type='GPU' if use_gpu else 'CPU',
		verbose=100
	)
	start_t = time.time()
	model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=40)
	elapsed = time.time() - start_t
	print(f'Fold {fold} 학습 시간: {elapsed/60:.1f}분')

	val_proba = model.predict_proba(X_val)[:,1]
	oof_preds[val_idx] = val_proba

	# 임계값 다변화 탐색
	thresholds = np.arange(0.01,0.99,0.02)
	recs = []
	for t in thresholds:
		y_bin = (val_proba >= t).astype(int)
		prec = precision_score(y_val, y_bin, zero_division=0)
		rec = recall_score(y_val, y_bin, zero_division=0)
		f1 = f1_score(y_val, y_bin, zero_division=0)
		tp = ((y_val==1)&(y_bin==1)).sum(); fn=((y_val==1)&(y_bin==0)).sum(); fp=((y_val==0)&(y_bin==1)).sum(); tn=((y_val==0)&(y_bin==0)).sum()
		tpr = tp/(tp+fn) if (tp+fn)>0 else 0
		fpr = fp/(fp+tn) if (fp+tn)>0 else 0
		youden = tpr - fpr
		recs.append({'threshold':t,'precision':prec,'recall':rec,'f1':f1,'youden':youden})
	thr_df = pd.DataFrame(recs)
	best_f1_thr = float(thr_df.loc[thr_df['f1'].idxmax()].threshold)
	cand = thr_df[thr_df['precision']>=0.08]
	if cand.empty:
		best_rp_thr = best_f1_thr
		best_rp_row = thr_df.loc[thr_df['f1'].idxmax()]
	else:
		best_rp_row = cand.sort_values('recall', ascending=False).iloc[0]
		best_rp_thr = float(best_rp_row.threshold)
	best_youden_thr = float(thr_df.loc[thr_df['youden'].idxmax()].threshold)
	final_thr = best_rp_thr if not cand.empty else best_f1_thr if thr_df.loc[thr_df['f1'].idxmax()].f1 >= thr_df.loc[thr_df['youden'].idxmax()].f1 else best_youden_thr

	fold_metrics.append({
		'fold': fold,
		'best_f1_thr': best_f1_thr,
		'best_rp_thr': best_rp_thr,
		'best_youden_thr': best_youden_thr,
		'final_thr': final_thr,
		'val_recall_final': recall_score(y_val,(val_proba>=final_thr).astype(int),zero_division=0),
		'val_precision_final': precision_score(y_val,(val_proba>=final_thr).astype(int),zero_division=0),
		'val_f1_final': f1_score(y_val,(val_proba>=final_thr).astype(int),zero_division=0)
	})
	models.append(model)

# OOF 임계값 최종 결정
print('\n[4] 전체 OOF 임계값 최종 결정')
thresholds = np.arange(0.01,0.99,0.01)
recs = []
for t in thresholds:
	y_bin = (oof_preds >= t).astype(int)
	prec = precision_score(y_all, y_bin, zero_division=0)
	rec = recall_score(y_all, y_bin, zero_division=0)
	f1 = f1_score(y_all, y_bin, zero_division=0)
	tp = ((y_all==1)&(y_bin==1)).sum(); fn=((y_all==1)&(y_bin==0)).sum(); fp=((y_all==0)&(y_bin==1)).sum(); tn=((y_all==0)&(y_bin==0)).sum()
	tpr = tp/(tp+fn) if (tp+fn)>0 else 0
	fpr = fp/(fp+tn) if (fp+tn)>0 else 0
	youden = tpr - fpr
	recs.append({'threshold':t,'precision':prec,'recall':rec,'f1':f1,'youden':youden})
full_thr_df = pd.DataFrame(recs)
full_thr_df.to_csv('output/thresholds_oof.csv', index=False)

best_f1_thr = float(full_thr_df.loc[full_thr_df['f1'].idxmax()].threshold)
candidate = full_thr_df[full_thr_df['precision']>=0.08]
if candidate.empty:
	best_rp_thr = best_f1_thr
	best_rp_row = full_thr_df.loc[full_thr_df['f1'].idxmax()]
else:
	best_rp_row = candidate.sort_values('recall', ascending=False).iloc[0]
	best_rp_thr = float(best_rp_row.threshold)
best_youden_thr = float(full_thr_df.loc[full_thr_df['youden'].idxmax()].threshold)
final_thr = best_rp_thr if not candidate.empty else best_f1_thr if full_thr_df.loc[full_thr_df['f1'].idxmax()].f1 >= full_thr_df.loc[full_thr_df['youden'].idxmax()].f1 else best_youden_thr
print(f'OOF 최종 선택 임계값: {final_thr:.3f}')

pd.DataFrame(fold_metrics).to_csv('output/cv_fold_metrics.csv', index=False)
with open('output/models/cv_preprocessor.pkl','wb') as f:
	pickle.dump(pre,f)
with open('output/models/cv_models.pkl','wb') as f:
	pickle.dump(models,f)
with open('output/models/cv_final_threshold.pkl','wb') as f:
	pickle.dump(final_thr,f)

print('\n✅ 교차검증 완료')
print('fold_metrics -> output/cv_fold_metrics.csv')
print('임계값 탐색 테이블 -> output/thresholds_oof.csv')
print('모델/전처리기 저장 완료')
PY

echo "\n다음 단계: bash run_cv_predict.sh 로 테스트 앙상블 제출 생성"
