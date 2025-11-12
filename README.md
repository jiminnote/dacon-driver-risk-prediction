

## 📌 프로젝트 개요

### 대회 정보
- **대회명**: Dacon 운수종사자 교통사고 위험 예측 AI 경진대회
- **목표**: 운전적성검사(A/B 검사) 결과를 기반으로 교통사고 위험 예측
- **평가지표**: ROC AUC
- **기간**: 2025년

### 문제 정의
운전적성검사 데이터를 활용하여 **교통사고 위험군을 사전에 식별**하고, 사고 예방을 위한 AI 모델 개발

---

## 데이터 분석

### 데이터 구조

| 구분 | 크기 | 컬럼 수 | 특징 |
|------|------|---------|------|
| **Train** | 944,767건 | - | 전체 학습 데이터 |
| **A 검사** | 647,241건 | 37개 | 신규 자격 취득자 대상 |
| **B 검사** | 297,526건 | 31개 | 자격 유지자 대상 |
| **Test** | 10건 | - | 제출용 예측 대상 |

### 주요 특징

#### 1. 극심한 클래스 불균형
```
정상군 (Label=0): 917,487건 (97.11%)
위험군 (Label=1):  27,280건 ( 2.89%)
                   ↑
              불균형 비율: 약 1:34
```
<img width="2637" height="973" alt="image" src="https://github.com/user-attachments/assets/5dc7f4e2-61f3-4b55-b594-16d8a9532daf" />
#### 2. 복잡한 컬럼 구조 및 컬럼명 정규화
- **원본 하이픈 패턴**: `A1-1`, `A1-2`, `A1-3`, `A1-4` → 의미 파악 어려움
- **정규화된 컬럼명**: `A1_Direction`, `A1_Speed`, `A1_Response`, `A1_ResponseTime`
  - A 검사: 32개 컬럼 rename (예: `A3-7` → `A3_ResponseTime`)
  - B 검사: 26개 컬럼 rename (예: `B2-2` → `B2_ResponseTime`)
- **시퀀스 데이터**: 반응시간 리스트 `"0.5,0.6,0.7,0.8"`
- **스칼라 데이터**: 정확도, 점수 등 단일 값
- **혼합 타입**: 컬럼별로 시퀀스/스칼라 혼재

#### 3. 데이터 품질 이슈
- 결측치 존재
- 상수 컬럼 (변별력 없음)
- Train/Test 피처 불일치 위험

---

### 피처 엔지니어링 전략

#### 1단계: 컬럼명 정규화
```python
# 원본 → 의미있는 이름으로 변환
RENAME_MAP_A = {
    'A1-1': 'A1_Direction',      # 방향 정보
    'A1-2': 'A1_Speed',          # 속도 정보
    'A1-3': 'A1_Response',       # 반응 결과
    'A1-4': 'A1_ResponseTime',   # 반응 시간
    'A3-7': 'A3_ResponseTime',   # 시각 탐색 반응시간
    'A4-5': 'A4_ResponseTime',   # 주의 전환 반응시간
    # ... 총 32개 컬럼
}

RENAME_MAP_B = {
    'B1-1': 'B1_Response1',      # 과제1 반응
    'B1-2': 'B1_ResponseTime',   # 반응 시간
    'B2-3': 'B2_Response2',      # 과제2 반응
    # ... 총 26개 컬럼
}
```

#### 2단계: 심리학적 의미 기반 피처 생성
```python
# A 검사: 인지 능력 측정 (44개 피처)
A_FEATURE_COLUMNS = [
    # A1: 주의 분배 능력
    'A1_response_rate', 'A1_left_response_rate', 'A1_right_response_rate',
    'A1_mean_response_time', 'A1_direction_diff_rt',
    
    # A2: 속도 지각 능력
    'A2_response_rate', 'A2_slow_to_fast_rt_diff', 
    'A2_correct_ratio_by_speed', 'A2_mean_response_time',
    
    # A3: 시각 탐색 능력 (가장 중요한 피처군)
    'A3_valid_accuracy', 'A3_invalid_accuracy', 'A3_total_accuracy',
    'A3_valid_rt', 'A3_invalid_rt', 'A3_accuracy_gap',
    
    # A4: 주의 전환 능력 (일치/불일치 조건)
    'A4_congruent_accuracy', 'A4_incongruent_accuracy', 
    'A4_accuracy_gap', 'A4_rt_gap',
    
    # A5: 변화 탐지 능력
    'A5_accuracy_non_change', 'A5_accuracy_pos_change',
    'A5_accuracy_color_change', 'A5_accuracy_shape_change',
    
    # A6-A7: 판단 및 상황 대처 점수
    'A6_score', 'A6_zscore', 'A7_score', 'A7_zscore',
    
    # A8: 일관성 및 왜곡 지표
    'A8_distortion_score', 'A8_consistency_score',
    
    # A9: 정서 안정성 프로파일
    'A9_emotional_stability', 'A9_behavior_stability',
    'A9_reality_checking', 'A9_cognitive_agility', 'A9_stress_level'
]

# B 검사: 지속 주의 능력 측정 (35개 피처)
B_FEATURE_COLUMNS = [
    # B1-B2: 변화 탐지 과제 (2가지 버전)
    'B1_task1_accuracy', 'B1_task2_change_acc', 
    'B1_task2_accuracy_gap', 'B1_task2_mean_rt',
    
    # B3: 단순 반응 과제
    'B3_accuracy', 'B3_mean_rt',
    
    # B4: Stroop 효과 (일치/불일치)
    'B4_congruent_accuracy', 'B4_incongruent_accuracy',
    'B4_accuracy_gap', 'B4_rt_gap',
    
    # B5-B8: 다양한 반응 정확도
    'B5_accuracy', 'B6_accuracy', 'B7_accuracy', 'B8_accuracy',
    
    # B9-B10: 신호 탐지 이론 (Hit/Miss/FA/CR)
    'B9_aud_hit', 'B9_aud_miss', 'B9_aud_fa', 'B9_aud_cr',
    'B10_aud_hit', 'B10_aud_fa', 'B10_vis2_correct'
]
```

### 최종 피처 현황

| 검사 타입 | 생성 피처 수 | 주요 특징 |
|-----------|--------------|-----------|
| **A 검사** | 44개 | 인지 능력 (주의, 지각, 판단, 정서) |
| **B 검사** | 35개 | 지속 주의 능력 (반응 일관성) |
| **총 합계** | 79개 | 심리학적 의미 기반 설계 |

**핵심 피처군**:
- 반응 정확도 (Accuracy)
- 반응 시간 (Response Time)
- 일치/불일치 조건 격차 (Gap)
- 변화 탐지 능력
- 정서 안정성 지표

---

## 성능 최적화 및 설계 전략

### 1. 컬럼명 정규화: 가독성과 의미 명확화

**문제점**: 
- 하이픈 패턴 (`A1-1`, `B2-3` 등) 의미 파악 어려움
- 피처 중요도 해석 시 심리학적 의미 불명확

**해결책**
```
RENAME_MAP_A = {
    'A1-1': 'A1_Direction',      # 주의 방향
    'A1-4': 'A1_ResponseTime',   # 반응 시간
    'A3-7': 'A3_ResponseTime',   # 시각 탐색 반응시간
    'A4-5': 'A4_ResponseTime',   # 주의 전환 반응시간
    'A6-1': 'A6_Count',          # 판단 능력 점수
}
```
### 2. 심리학적 의미 기반 피처 생성

**문제점**: 
- 초기 접근: 단순 통계 (mean, std, min, max...) → 257개 피처
- 결과: 피처 간 의미 중복, 해석 어려움

**개선된 접근**:
```python
# 심리학적 구인(construct) 기반 설계
# A3 검사: 시각 탐색 능력 측정
features = {
    'A3_valid_accuracy': valid_correct / valid_total,      # 유효 조건 정확도
    'A3_invalid_accuracy': invalid_correct / invalid_total, # 무효 조건 정확도
    'A3_accuracy_gap': abs(valid_acc - invalid_acc),       # 조건 간 격차
    'A3_valid_rt': mean(valid_response_times),             # 유효 조건 반응시간
    'A3_invalid_rt': mean(invalid_response_times),         # 무효 조건 반응시간
}

# A4 검사: Stroop 효과 측정
features = {
    'A4_congruent_accuracy': ...,      # 일치 조건 정확도
    'A4_incongruent_accuracy': ...,    # 불일치 조건 정확도
    'A4_accuracy_gap': ...,            # 조건 간 격차 (인지적 통제력)
    'A4_rt_gap': ...,                  # 반응시간 격차
}
```

**효과**:
- 257개 → 79개로 축소 (의미 중복 제거)
- 각 피처가 특정 인지 능력 반영
- 모델 해석력 향상

### 3. 벡터화를 통한 처리 속도 개선 

**초기 구현 (느림)**:
```python
# Row-level loop
for idx, val in df[col].items():
    seq = parse_sequence(val)
    features.loc[idx, f'{col}_mean'] = np.mean(seq)
    # ...
```
- **소요 시간**: A 검사 ~수 시간

**개선된 구현 (빠름)**:
```python
# Vectorized operations
seqs = col_series.apply(lambda s: parse_sequence(s))
new_cols_dict[f'{col}_mean'] = [np.mean(s) if s else np.nan for s in seqs]
features = pd.concat([features, pd.DataFrame(new_cols_dict)], axis=1)
```

**성능 개선 결과**:
```
처리 시간 비교:

(벡터화):
  A 검사: 415.6초 (~7분)   ⚡ 10배+ 향상
  B 검사: 108.4초 (~2분)   ⚡ 10배+ 향상

```

---

## 모델링 전략

### 1. 전처리 파이프라인 (FeaturePreprocessor)

**목적**: Train/Test 피처 정렬 일관성 보장

```
Train 단계:

1. 상수 컬럼 제거               
2. 유효값 < 10개 컬럼 제거     
3. 컬럼 순서 고정         
4. 중앙값 계산 및 저장          
              ↓
  preprocessor.pkl 저장

Test 단계:

1. Train 기준 컬럼만 선택  
2. 누락 컬럼 자동 추가 (NaN)   
3. Train 중앙값으로 결측 대체
4. 컬럼 순서 Train과 동일화

```

### 2. 불균형 처리

#### 방법 1: Auto Class Weights
```python
CatBoostClassifier(
    auto_class_weights='Balanced',  # 자동 가중치
    # 양성 클래스에 ~34배 가중치 부여
)
```

#### 방법 2: Threshold 최적화 (다중 기준)
<img width="3568" height="1768" alt="image" src="https://github.com/user-attachments/assets/243fe5f2-a34d-425a-97a7-8e72c8af9b79" />

```
Threshold Sweep (0.01 ~ 0.99):
<img width="435" height="140" alt="스크린샷 2025-11-12 오후 5 50 31" src="https://github.com/user-attachments/assets/1806a3c6-1577-4799-b225-4cc552236cd7" />


선택 기준:
1순위: Recall@Precision≥0.08
2순위: F1 Score
3순위: Youden's J (TPR - FPR)
```

### 3. 모델 구성
<img width="1790" height="490" alt="image" src="https://github.com/user-attachments/assets/55ac5904-8408-477a-9b04-546825cc5ef7" />
#### CatBoost
<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/e4d17619-301f-4526-9007-6c16ba52541a" />
#### XGBoost Baseline
<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/3bc62538-8ebd-41ec-98dc-a27e32416008" />
#### LightGBM
<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/60446af5-26bd-4ec3-82c5-cd7c09232263" />

<img width="1489" height="1189" alt="image" src="https://github.com/user-attachments/assets/bb2d8c52-81bc-403d-a3df-cb9060e4a349" />
<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/1326b5dc-6b81-4568-adf7-8581e30d93cb" />

Base Models (각 5-Fold):
<img width="283" height="242" alt="스크린샷 2025-11-12 오후 5 50 44" src="https://github.com/user-attachments/assets/96d2648b-f964-46a9-9f3a-78b3da107a7b" />

```

---

## 모델 성능 비교

### 최종 성과 요약

<img width="2969" height="1768" alt="image" src="https://github.com/user-attachments/assets/0cf0280b-b473-4b3c-a8d3-1f4b2df4c561" />

| 모델 | OOF ROC AUC | 특징 | 
|------|-------------|------|
| Baseline | 0.6384 | 단일 CatBoost + 최적 임계값 | 
| CV Mean | 0.6381 | 5-Fold 평균 (안정적) |
| CV Weighted | 0.6381 | F1 가중 평균 | 
| Feature Selection | 0.6386 | Top-150 피처만 사용 | 
| **Stacking** | **0.6385** | 멀티모델 앙상블 | 

### 성능 개선 여정

```
단계별 ROC AUC 변화:

초기 (피처 추출 실패)
│  ROC AUC: 0.5678
│  문제: 하이픈 패턴 미인식, 피처 8개만 생성
│
▼ [1단계] 하이픈 패턴 스캔 로직 추가
│
│  ROC AUC: 0.6384  (+0.0706 ↑)
│  개선: 257개 피처 생성 성공
│
▼ [2단계] 벡터화 최적화
│
│  ROC AUC: 0.6384 (유지)
│  개선: 처리 시간 10배+ 단축
│
▼ [3단계] 임계값 다변화
│
│  ROC AUC: 0.6384 (유지)
│  개선: Recall 0.0 → 0.218 향상
│
▼ [4단계] Cross-Validation
│
│  ROC AUC: 0.6381
│  개선: 일반화 성능, 안정성 확보
│
▼ [5단계] Stacking Ensemble
│
│  ROC AUC: 0.6385 (+0.0001 ↑)
│  최종: 멀티모델 다양성 확보
│
```

### 성능 지표 상세

```
최종 모델 (Stacking) 성능:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROC AUC:      0.6385
Accuracy:     0.8701
Precision:    0.0555  (임계값 0.5 기준)
Recall:       0.2181  (임계값 0.5 기준)
F1 Score:     0.0884
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

최적 임계값 (0.04) 적용 시:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recall:       0.8920  (↑)
Precision:    0.0412  (↓)
F1 Score:     0.0789
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 시각화 자료
<img width="2969" height="1768" alt="image" src="https://github.com/user-attachments/assets/1d4a1ac7-028c-4154-9c11-4c7dc939639f" />

### 1. 피처 중요도 (Top 20)
Feature Importance (평균 - 5 Folds):
<img width="2970" height="2968" alt="image" src="https://github.com/user-attachments/assets/e71b5067-d7e7-41ec-8d75-57ef19ece7b0" />

**주요 인사이트:**
A3_ResponseTime (시각 탐색 반응시간)이 가장 중요
A6_zscore (판단 능력 표준점수) 중요
Gap 피처 (조건 간 격차) 상위권
일관성 지표가 사고 위험 예측에 효과적


### 2. 클래스 분포

```
Train 데이터 클래스 분포:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Label = 0 (정상):  917,487건 (97.11%)
Label = 1 (위험):   27,280건 (2.89%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
불균형 비율: 1:34
→ 불균형 처리 필수 (Auto Class Weights, Threshold 최적화)
```

---

## 🚀실행 파이프라인

### 전체 워크플로우

```
1. Baseline 학습

   bash run_pipeline.sh 

              ↓
   submission.csv 생성

2. 교차검증 앙상블

   bash run_cv_pipeline.sh

              ↓
   cv_models.pkl, cv_preprocessor.pkl 저장
              ↓

   bash run_cv_predict.sh 

              ↓
   submission_cv_mean.csv
   submission_cv_weighted.csv 생성

3. Feature Selection

 bash run_feature_selection.sh

              ↓
   feature_importance_mean.csv
   feature_selected_model.pkl 저장

4. 멀티모델 스태킹

   bash run_stacking_pipeline.sh

              ↓
   stack_*_models.pkl, stack_meta_model.pkl 저장
              ↓
 
   bash run_stacking_predict.sh

              ↓
   submission_stacking.csv 생성 
```

## 💡 주요 인사이트 & 기술적 성과

### 1. 데이터 구조 이해의 중요성
- 초기: 컬럼명 패턴 오해, 단순 통계 피처만 생성
- **개선 1단계**: 하이픈 패턴 인식 → 컬럼명 정규화
- **개선 2단계**: 심리학적 의미 기반 피처 설계
- 결과: ROC AUC **+0.07** 향상, 해석력 증가

### 2. 피처 설계 철학
- **단순 통계 → 도메인 지식**: 심리학적 구인 반영
- **양보다 질**: 257개 → 79개 (의미 있는 피처만)
- **해석 가능성**: 각 피처가 특정 인지 능력 측정

### 3. 성능 최적화
- 벡터화로 피처 생성 시간 **10배 이상** 단축
- 실험 반복 주기 단축 → 빠른 개선 사이클

### 3. 불균형 처리
- Auto Class Weights: 모델 학습 단계 보정
- Threshold 최적화: 예측 단계 보정
- 이중 전략으로 Recall **0.0 → 0.22** 향상

### 5. 앙상블의 한계
- Stacking 개선폭 미미 (+0.0001): Base 모델 다양성 부족
- CV Mean과 유사한 성능: 안정성 확보

### 6. 주요 피처 인사이트
- **A3_ResponseTime**: 시각 탐색 반응시간이 최고 중요도
- **A6_zscore**: 판단 능력 표준점수 상위권
- **Gap 피처**: 조건 간 격차가 인지적 통제력 반영
- **일관성 지표**: 반응시간 변동성이 사고 위험 예측

### 7. Feature Selection 효과
- 79개 전체 사용으로 충분한 성능 확보
- 도메인 지식 기반 설계로 불필요한 피처 사전 제거
- 과적합 위험 감소, 추론 속도 향상

---

## 최종 제출 전략

### 제출 파일 비교

| 순위 | 파일명 | 모델 | 기대 성능 | 특징 |
|------|--------|------|-----------|------|
| 1 | `submission_stacking.csv` | 멀티모델 스태킹 | 0.6385 | 최고 성능, 다양성 확보 |
| 2 | `submission_cv_mean.csv` | CV 5-Fold 평균 | 0.6381 | 안정적, 일반화 우수 |
| 3 | `submission.csv` | Baseline | 0.6384 | 단순, 빠른 추론 |
| 4 | `submission_cv_weighted.csv` | CV F1 가중 | 0.6381 | Mean과 유사 |

<img width="1455" height="906" alt="image" src="https://github.com/user-attachments/assets/13fd32b7-2234-4d3d-a908-dd70dab00976" />
---

## 향후 개선 방향

### 1. 추가 피처 엔지니어링
- **교차 피처**: A × B 검사 상호작용 (예: `A3_ResponseTime * B4_rt_gap`)
- **비율 피처**: 정확도 대비 반응시간 효율성
- **시계열 피처**: 검사 날짜 기반 트렌드 (있다면)
- **군집 기반 피처**: 반응 패턴 유사도

### 2. 앙상블 다양성 확보
- **다른 알고리즘**: Neural Network, SVM
- **다른 피처 부분집합**: Random Subspace
- **다른 샘플링**: Bagging, Boosting 변형

### 3. 하이퍼파라미터 최적화
- **Optuna**: 자동 튜닝
- **Grid/Random Search**: 체계적 탐색
- **Early Stopping**: 최적 iteration 탐색

### 4. 외부 데이터 활용
- **교통사고 통계**: 지역/시간대별 사고율
- **기상 데이터**: 날씨와 사고 상관관계
- **인구통계**: 연령/성별 위험도

### 5. 모델 해석
- **SHAP Values**: 예측 설명력 향상
- **Partial Dependence**: 피처 영향 분석
- **Feature Interaction**: 상호작용 발견


## 📝 결론

### 프로젝트 성과
극심한 불균형 문제 해결 (1:34 비율)  
복잡한 데이터 구조 파싱 및 컬럼명 정규화 성공  
심리학적 의미 기반 피처 설계 (79개 핵심 피처)  
성능 최적화 달성 (피처 생성 10배+ 속도 향상)  
경쟁력 있는 모델 구축 (ROC AUC 0.6385)  
해석 가능한 모델 (피처 중요도 분석)  
재현 가능한 파이프라인 완성  


## 📚 기술 스택

### 프로그래밍 & 프레임워크

```
Python 3.x
├─ pandas, numpy          (데이터 처리)
├─ scikit-learn           (전처리, 검증, 메타 모델)
├─ CatBoost               (주 모델)
├─ LightGBM, XGBoost      (앙상블)
├─ imbalanced-learn       (SMOTE)
└─ matplotlib, seaborn    (시각화)
```

---

## 👥 프로젝트 정보
 
**기간**: 2025년  
**대회**: Dacon 운수종사자 교통사고 위험 예측 AI 경진대회  
**평가지표**: ROC AUC  
**최종 성과**: ROC AUC 0.6385  

---
