"""
공통 유틸리티 함수
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

# 프로젝트 루트 디렉토리 추정 (이 파일이 utils/ 아래에 있다고 가정)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_data(data_path=None):
    """데이터 로드

    - data_path가 None이면 프로젝트 루트의 data 폴더 사용
    - 호출 위치와 무관하게 절대 경로로 접근
    """
    data_dir = Path(data_path) if data_path is not None else (PROJECT_ROOT / 'data')

    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    train_a = pd.read_csv(data_dir / 'train' / 'A.csv')
    train_b = pd.read_csv(data_dir / 'train' / 'B.csv')
    test_a = pd.read_csv(data_dir / 'test' / 'A.csv')
    test_b = pd.read_csv(data_dir / 'test' / 'B.csv')
    
    return train, test, train_a, train_b, test_a, test_b

def print_data_info(df, name='DataFrame'):
    """데이터 기본 정보 출력"""
    print(f"\n{'='*60}")
    print(f"{name} 정보")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nBasic Statistics:")
    print(df.describe())

def evaluate_model(y_true, y_pred_proba, threshold=0.5, model_name='Model'):
    """모델 평가 및 지표 출력"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'model': model_name,
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    print(f"\n{'='*60}")
    print(f"{model_name} 평가 결과 (Threshold: {threshold})")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if key != 'model':
            print(f"{key.upper():12s}: {value:.4f}")
    
    return metrics

def plot_roc_pr_curves(y_true, y_pred_proba, model_name='Model', save_path=None):
    """ROC 및 PR Curve 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    axes[0].plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    axes[1].plot(recall, precision, lw=2, label=model_name)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name='Model', save_path=None):
    """Confusion Matrix 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['비위험군', '위험군'],
                yticklabels=['비위험군', '위험군'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Feature Importance 시각화"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    else:
        print("모델에서 Feature Importance를 추출할 수 없습니다.")
        return
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return importance_df

def save_submission(test_ids, predictions, file_path=None):
    """제출 파일 생성

    - file_path가 None이면 프로젝트 루트의 output/submissions/submission.csv로 저장
    - 저장 경로가 없으면 자동 생성
    """
    submission = pd.DataFrame({
        'Test_id': test_ids,
        'Label': predictions
    })
    submission = submission.sort_values('Test_id').reset_index(drop=True)

    target_path = Path(file_path) if file_path is not None else (PROJECT_ROOT / 'output' / 'submissions' / 'submission.csv')
    _ensure_dir(target_path.parent)
    submission.to_csv(target_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"제출 파일 저장 완료: {target_path}")
    print(f"{'='*60}")
    print(f"Shape: {submission.shape}")
    print(f"Label 범위: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Label 평균: {predictions.mean():.4f}")
    print(f"\n샘플:")
    print(submission.head(10))
    
    return submission
