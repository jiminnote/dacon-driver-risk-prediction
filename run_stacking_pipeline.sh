#!/bin/bash
# 멀티모델 스태킹 파이프라인: CatBoost + LightGBM + XGBoost 5-Fold OOF → 메타 로지스틱 회귀
# 사용: bash run_stacking_pipeline.sh [TOP_N(optional)]
# TOP_N 주면 feature_selection 결과 상위 피처만 사용 (없으면 전체)

set -e
TOP_N=${1:-0}

echo "🚀 멀티모델 스태킹 파이프라인 시작 (TOP_N=${TOP_N})"

python <<PY
import os, pickle, numpy as np, pandas as pd, time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

import sys
sys.path.append('utils'); sys.path.append('features')
from common_utils import load_data, evaluate_model
from feature_engineering import build_a_features, build_b_features, merge_features_with_labels
from preprocessing import FeaturePreprocessor

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

print('[1] 데이터 로드 & 피처 생성')
train, test, train_a, train_b, test_a, test_b = load_data()
fa = build_a_features(train_a)
fb = build_b_features(train_b)
X_raw, y = merge_features_with_labels(train, fa, fb)

print('[2] 전처리기 적합')
pre = FeaturePreprocessor().fit(X_raw)
X_full = pre.transform(X_raw)
features_all = X_full.columns.tolist()

if ${TOP_N} > 0 and os.path.exists('output/feature_importance_mean.csv'):
    imp_df = pd.read_csv('output/feature_importance_mean.csv')
    selected = imp_df.head(int(${TOP_N}))['feature'].tolist()
    # 선택된 컬럼이 전처리 결과에 없는 경우 필터링
    selected = [c for c in selected if c in features_all]
    X_full = X_full[selected]
    print(f'TOP_N={${TOP_N}} 적용 후 Shape: {X_full.shape}')
else:
    print('전체 피처 사용')

K=5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

print('[3] Base 모델 OOF 생성')
cat_oof = np.zeros(len(y)); lgb_oof = np.zeros(len(y)); xgb_oof = np.zeros(len(y))
cat_models=[]; lgb_models=[]; xgb_models=[]

for fold,(tr_idx,val_idx) in enumerate(skf.split(X_full,y),1):
    print(f'\n▶ Fold {fold}/{K}')
    X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    cat = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, auto_class_weights='Balanced', random_seed=fold+10, verbose=100)
    cat.fit(X_tr,y_tr, eval_set=(X_val,y_val), early_stopping_rounds=40)
    cat_p = cat.predict_proba(X_val)[:,1]
    cat_oof[val_idx] = cat_p
    cat_models.append(cat)

    lgb = LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=fold+20, verbose=-1)
    lgb.fit(X_tr,y_tr, eval_set=[(X_val,y_val)], eval_metric='auc')
    lgb_p = lgb.predict_proba(X_val)[:,1]
    lgb_oof[val_idx] = lgb_p
    lgb_models.append(lgb)

    xgb = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=fold+30, eval_metric='auc', use_label_encoder=False, verbosity=0)
    xgb.fit(X_tr,y_tr, eval_set=[(X_val,y_val)])
    xgb_p = xgb.predict_proba(X_val)[:,1]
    xgb_oof[val_idx] = xgb_p
    xgb_models.append(xgb)

print('\nBase 모델 OOF AUC들:')
print(' CatBoost AUC:', roc_auc_score(y, cat_oof))
print(' LightGBM AUC:', roc_auc_score(y, lgb_oof))
print(' XGBoost  AUC:', roc_auc_score(y, xgb_oof))

print('[4] 메타 데이터 구성')
meta_X = pd.DataFrame({'cat':cat_oof,'lgb':lgb_oof,'xgb':xgb_oof})
meta_y = y.copy()

print('[5] 메타 로지스틱 회귀 학습 (5-Fold 내부 OOF 사용)')
# 간단히 전체 OOF로 학습 (Nested CV 생략)
stack_model = LogisticRegression(max_iter=1000)
stack_model.fit(meta_X, meta_y)
meta_proba = stack_model.predict_proba(meta_X)[:,1]
auc_stack = roc_auc_score(meta_y, meta_proba)
print(f'Stacking OOF AUC: {auc_stack:.4f}')

# 임계값 다변화 동일 로직 적용
thresholds = np.arange(0.01,0.99,0.01)
recs=[]
for t in thresholds:
    yb=(meta_proba>=t).astype(int)
    prec=precision_score(meta_y,yb,zero_division=0)
    rec=recall_score(meta_y,yb,zero_division=0)
    f1=f1_score(meta_y,yb,zero_division=0)
    tp=((meta_y==1)&(yb==1)).sum(); fn=((meta_y==1)&(yb==0)).sum(); fp=((meta_y==0)&(yb==1)).sum(); tn=((meta_y==0)&(yb==0)).sum()
    tpr=tp/(tp+fn) if (tp+fn)>0 else 0; fpr=fp/(fp+tn) if (fp+tn)>0 else 0
    youden=tpr-fpr
    recs.append({'threshold':t,'precision':prec,'recall':rec,'f1':f1,'youden':youden})
thr_df=pd.DataFrame(recs)
thr_df.to_csv('output/thresholds_stacking_oof.csv',index=False)

best_f1_thr=float(thr_df.loc[thr_df['f1'].idxmax()].threshold)
candidate=thr_df[thr_df['precision']>=0.08]
if candidate.empty:
    best_rp_thr=best_f1_thr
    best_rp_row=thr_df.loc[thr_df['f1'].idxmax()]
else:
    best_rp_row=candidate.sort_values('recall',ascending=False).iloc[0]
    best_rp_thr=float(best_rp_row.threshold)

best_youden_thr=float(thr_df.loc[thr_df['youden'].idxmax()].threshold)
final_thr=best_rp_thr if not candidate.empty else best_f1_thr if thr_df.loc[thr_df['f1'].idxmax()].f1 >= thr_df.loc[thr_df['youden'].idxmax()].f1 else best_youden_thr
print(f'Staking 최종 임계값: {final_thr:.3f}')

# 저장
with open('output/models/stack_preprocessor.pkl','wb') as f:
    pickle.dump(pre,f)
with open('output/models/stack_cat_models.pkl','wb') as f:
    pickle.dump(cat_models,f)
with open('output/models/stack_lgb_models.pkl','wb') as f:
    pickle.dump(lgb_models,f)
with open('output/models/stack_xgb_models.pkl','wb') as f:
    pickle.dump(xgb_models,f)
with open('output/models/stack_meta_model.pkl','wb') as f:
    pickle.dump(stack_model,f)
with open('output/models/stack_final_threshold.pkl','wb') as f:
    pickle.dump(final_thr,f)

pd.DataFrame([{'auc_cat':roc_auc_score(y,cat_oof),'auc_lgb':roc_auc_score(y,lgb_oof),'auc_xgb':roc_auc_score(y,xgb_oof),'auc_stack':auc_stack,'final_thr':final_thr,'top_n':${TOP_N}}]).to_csv('output/stacking_summary.csv',index=False)
print('\n✅ 스태킹 학습 완료: stacking_summary.csv 저장')
PY

echo "🎉 스태킹 파이프라인 완료 (stacking_summary.csv, 모델/임계값 저장)"
