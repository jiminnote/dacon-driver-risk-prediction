"""
FeaturePreprocessor: Train/Test 동일 컬럼 정렬 및 결측/상수 처리 일관성 보장
"""
from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List

class FeaturePreprocessor:
    def __init__(self):
        self.keep_columns: List[str] = []
        self.medians: Dict[str, float] = {}
        self.fitted: bool = False

    def _sanitize(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.replace([np.inf, -np.inf], np.nan)
        return X

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        X = self._sanitize(X.copy())
        # 상수 컬럼 제거 기준 계산 (train 기준)
        # 개선: 유효값이 최소 10개 이상 있을 때만 상수 여부 판단
        constant_cols = []
        for col in X.columns:
            valid_count = X[col].notna().sum()
            if valid_count < 10:
                # 유효값이 너무 적으면 일단 제거
                constant_cols.append(col)
            elif X[col].nunique(dropna=True) <= 1:
                # 유효값 중 고유값이 1개 이하면 상수
                constant_cols.append(col)
        
        self.keep_columns = [col for col in X.columns if col not in constant_cols]
        print(f"전처리: {len(constant_cols)}개 컬럼 제거 (유효값 부족 또는 상수), {len(self.keep_columns)}개 유지")
        
        # 중앙값 계산 (train 기준)
        med = X[self.keep_columns].median(numeric_only=True)
        self.medians = med.to_dict()
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted, "Preprocessor is not fitted. Call fit() first or load a fitted preprocessor."
        X = self._sanitize(X.copy())
        # 불필요한 컬럼 제거
        X = X[[col for col in X.columns if col in self.keep_columns]].copy()
        # 누락된 컬럼 추가 (train 기준 컬럼 모두 보장)
        for col in self.keep_columns:
            if col not in X.columns:
                X[col] = np.nan
        # 컬럼 순서 정렬 (train 기준 순서와 동일하게)
        X = X[self.keep_columns]
        # 결측치 채움 (train 중앙값)
        for col, med in self.medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(med)
        # 남은 결측은 0으로 보수 (안정성)
        if X.isnull().any().any():
            X = X.fillna(0)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'keep_columns': self.keep_columns,
                'medians': self.medians,
                'fitted': self.fitted,
            }, f)

    @staticmethod
    def load(path: str | Path) -> "FeaturePreprocessor":
        with open(path, 'rb') as f:
            state = pickle.load(f)
        # 호환성: 과거에는 전체 객체 FeaturePreprocessor를 직접 pickle.dump 했을 수 있음
        if isinstance(state, FeaturePreprocessor):
            return state
        # dict 형태 저장 (현재 save 방식)
        if isinstance(state, dict):
            obj = FeaturePreprocessor()
            obj.keep_columns = state.get('keep_columns', [])
            obj.medians = state.get('medians', {})
            obj.fitted = state.get('fitted', False)
            return obj
        raise TypeError(f"Unsupported preprocessor state type: {type(state)}")
