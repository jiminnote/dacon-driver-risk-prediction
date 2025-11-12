import numpy as np
import pandas as pd

RENAME_MAP_A = {
    'A1-1': 'A1_Direction',
    'A1-2': 'A1_Speed',
    'A1-3': 'A1_Response',
    'A1-4': 'A1_ResponseTime',
    'A2-1': 'A2_Speed1',
    'A2-2': 'A2_Speed2',
    'A2-3': 'A2_Response',
    'A2-4': 'A2_ResponseTime',
    'A3-1': 'A3_ArrowSize',
    'A3-2': 'A3_ArrowPosition',
    'A3-3': 'A3_ArrowDirection',
    'A3-4': 'A3_CorrectPosition',
    'A3-5': 'A3_ResponseType',
    'A3-6': 'A3_Response',
    'A3-7': 'A3_ResponseTime',
    'A4-1': 'A4_Condition',
    'A4-2': 'A4_Color',
    'A4-3': 'A4_Response1',
    'A4-4': 'A4_Response2',
    'A4-5': 'A4_ResponseTime',
    'A5-1': 'A5_ChangeType',
    'A5-2': 'A5_Response1',
    'A5-3': 'A5_Response',
    'A6-1': 'A6_Count',
    'A7-1': 'A7_Count',
    'A8-1': 'A8_Count1',
    'A8-2': 'A8_Count2',
    'A9-1': 'A9_Count1',
    'A9-2': 'A9_Count2',
    'A9-3': 'A9_Count3',
    'A9-4': 'A9_Count4',
    'A9-5': 'A9_Count5'
}

RENAME_MAP_B = {
    'B1-1': 'B1_Response1',
    'B1-2': 'B1_ResponseTime',
    'B1-3': 'B1_Response2',
    'B2-1': 'B2_Response1',
    'B2-2': 'B2_ResponseTime',
    'B2-3': 'B2_Response2',
    'B3-1': 'B3_Response',
    'B3-2': 'B3_ResponseTime',
    'B4-1': 'B4_Response',
    'B4-2': 'B4_ResponseTime',
    'B5-1': 'B5_Response',
    'B5-2': 'B5_ResponseTime',
    'B6': 'B6_Response',
    'B7': 'B7_Response',
    'B8': 'B8_Response',
    'B9-1': 'B9_Count1',
    'B9-2': 'B9_Count2',
    'B9-3': 'B9_Count3',
    'B9-4': 'B9_Count4',
    'B9-5': 'B9_Count5',
    'B10-1': 'B10_Count1',
    'B10-2': 'B10_Count2',
    'B10-3': 'B10_Count3',
    'B10-4': 'B10_Count4',
    'B10-5': 'B10_Count5',
    'B10-6': 'B10_Count6'
}

A_FEATURE_COLUMNS = [
    'A1_response_rate', 'A1_left_response_rate', 'A1_right_response_rate', 'A1_fast_response_rate',
    'A1_mean_response_time', 'A1_fast_avg_rt', 'A1_direction_diff_rt',
    'A2_response_rate', 'A2_slow_to_fast_rt_diff', 'A2_correct_ratio_by_speed', 'A2_mean_response_time',
    'A3_valid_accuracy', 'A3_invalid_accuracy', 'A3_total_accuracy', 'A3_valid_rt', 'A3_invalid_rt',
    'A3_correct_rt', 'A3_incorrect_rt', 'A3_accuracy_gap',
    'A4_congruent_accuracy', 'A4_incongruent_accuracy', 'A4_accuracy_gap', 'A4_mean_rt_con',
    'A4_mean_rt_incon', 'A4_rt_gap', 'A4_response_rate',
    'A5_accuracy_non_change', 'A5_accuracy_pos_change', 'A5_accuracy_color_change', 'A5_accuracy_shape_change',
    'A5_accuracy_var',
    'A6_score', 'A6_zscore', 'A7_score', 'A7_zscore',
    'A8_distortion_score', 'A8_consistency_score', 'A8_distortion_flag',
    'A9_emotional_stability', 'A9_behavior_stability', 'A9_reality_checking',
    'A9_cognitive_agility', 'A9_stress_level', 'A9_total_score', 'A9_stability_gap'
]

B_FEATURE_COLUMNS = [
    'B1_task1_accuracy', 'B1_task2_change_acc', 'B1_task2_non_change_acc', 'B1_task2_accuracy_gap', 'B1_task2_mean_rt',
    'B2_task1_accuracy', 'B2_task2_change_acc', 'B2_task2_non_change_acc', 'B2_task2_accuracy_gap', 'B2_task2_mean_rt',
    'B3_accuracy', 'B3_mean_rt',
    'B4_congruent_accuracy', 'B4_incongruent_accuracy', 'B4_accuracy_gap', 'B4_mean_rt_congruent', 'B4_mean_rt_incongruent', 'B4_rt_gap',
    'B5_accuracy', 'B5_mean_rt',
    'B6_accuracy', 'B7_accuracy', 'B8_accuracy',
    'B9_aud_hit', 'B9_aud_miss', 'B9_aud_fa', 'B9_aud_cr', 'B9_vis_err',
    'B10_aud_hit', 'B10_aud_miss', 'B10_aud_fa', 'B10_aud_cr', 'B10_vis1_err', 'B10_vis2_correct'
]

def rename_a_columns(df):
    columns = {k: v for k, v in RENAME_MAP_A.items() if k in df.columns}
    return df.rename(columns=columns)

def rename_b_columns(df):
    columns = {k: v for k, v in RENAME_MAP_B.items() if k in df.columns}
    return df.rename(columns=columns)

def parse_sequence(value, dtype=float):
    if isinstance(value, str) and value not in ('', 'nan', 'NaN'):
        arr = np.fromstring(value, sep=',')
        if arr.size == 0:
            return np.array([], dtype=np.float64 if dtype is float else np.int64)
        return arr.astype(np.float64 if dtype is float else np.int64)
    return np.array([], dtype=np.float64 if dtype is float else np.int64)

def parse_int_sequence(value):
    return parse_sequence(value, dtype=int)

def parse_float_sequence(value):
    return parse_sequence(value, dtype=float)

def align_sequences(*arrays):
    lengths = [arr.size for arr in arrays if arr.size > 0]
    if not lengths:
        return arrays
    min_len = min(lengths)
    aligned = []
    for arr in arrays:
        if arr.size and arr.size != min_len:
            aligned.append(arr[:min_len])
        else:
            aligned.append(arr)
    return tuple(aligned)

def safe_mean(array):
    if array.size == 0:
        return np.nan
    if np.all(np.isnan(array)):
        return np.nan
    return float(np.nanmean(array))

def safe_rate(array):
    return safe_mean(array)

def safe_diff(val1, val2):
    if np.isnan(val1) or np.isnan(val2):
        return np.nan
    return float(val1 - val2)

def mean_ignore_zeros(array):
    if array.size == 0:
        return np.nan
    non_zero = array[array != 0]
    if non_zero.size:
        return safe_mean(non_zero)
    return safe_mean(array)

def binary_accuracy(seq, success_code=1, failure_code=2):
    if seq.size == 0:
        return np.nan
    mask = np.isin(seq, [success_code, failure_code])
    if not mask.any():
        return np.nan
    values = np.where(seq[mask] == success_code, 1.0, 0.0)
    return safe_mean(values)

def compute_a1_features(row):
    direction = parse_int_sequence(row.get('A1_Direction'))
    speed = parse_int_sequence(row.get('A1_Speed'))
    responses = parse_int_sequence(row.get('A1_Response'))
    rt = parse_float_sequence(row.get('A1_ResponseTime'))
    direction, speed, responses, rt = align_sequences(direction, speed, responses, rt)
    feats = {
        'A1_response_rate': safe_rate(responses),
        'A1_left_response_rate': safe_rate(responses[direction == 1]) if direction.size else np.nan,
        'A1_right_response_rate': safe_rate(responses[direction == 2]) if direction.size else np.nan,
        'A1_fast_response_rate': safe_rate(responses[speed == 3]) if speed.size else np.nan,
        'A1_mean_response_time': safe_mean(rt)
    }
    feats['A1_fast_avg_rt'] = safe_mean(rt[speed == 3]) if speed.size else np.nan
    left_rt = safe_mean(rt[direction == 1]) if direction.size else np.nan
    right_rt = safe_mean(rt[direction == 2]) if direction.size else np.nan
    feats['A1_direction_diff_rt'] = safe_diff(left_rt, right_rt)
    return feats

def compute_a2_features(row):
    speed1 = parse_int_sequence(row.get('A2_Speed1'))
    speed2 = parse_int_sequence(row.get('A2_Speed2'))
    responses = parse_int_sequence(row.get('A2_Response'))
    rt = parse_float_sequence(row.get('A2_ResponseTime'))
    speed1, speed2, responses, rt = align_sequences(speed1, speed2, responses, rt)
    feats = {'A2_response_rate': safe_rate(responses), 'A2_mean_response_time': safe_mean(rt)}
    if rt.size:
        slow_to_fast_mask = (speed1 < speed2)
        fast_to_slow_mask = (speed1 > speed2)
        slow_fast_rt = safe_mean(rt[slow_to_fast_mask]) if slow_to_fast_mask.any() else np.nan
        fast_slow_rt = safe_mean(rt[fast_to_slow_mask]) if fast_to_slow_mask.any() else np.nan
        feats['A2_slow_to_fast_rt_diff'] = safe_diff(slow_fast_rt, fast_slow_rt)
    else:
        feats['A2_slow_to_fast_rt_diff'] = np.nan
    speeds = np.unique(speed2[speed2 > 0]) if speed2.size else np.array([])
    if speeds.size >= 2:
        accs = [safe_rate(responses[speed2 == s]) for s in speeds]
        feats['A2_correct_ratio_by_speed'] = safe_diff(accs[-1], accs[0])
    else:
        feats['A2_correct_ratio_by_speed'] = np.nan
    return feats

def compute_a3_features(row):
    rtype = parse_int_sequence(row.get('A3_ResponseType'))
    responses = parse_int_sequence(row.get('A3_Response'))
    rt = parse_float_sequence(row.get('A3_ResponseTime'))
    rtype, responses, rt = align_sequences(rtype, responses, rt)
    feats = {}
    if rtype.size:
        valid_mask = np.isin(rtype, [1, 2])
        invalid_mask = np.isin(rtype, [3, 4])
        valid_correct = (rtype == 1).sum()
        valid_total = valid_mask.sum()
        invalid_correct = (rtype == 3).sum()
        invalid_total = invalid_mask.sum()
        total_correct = valid_correct + invalid_correct
        total_trials = rtype.size
        feats['A3_valid_accuracy'] = (valid_correct / valid_total) if valid_total else np.nan
        feats['A3_invalid_accuracy'] = (invalid_correct / invalid_total) if invalid_total else np.nan
        feats['A3_total_accuracy'] = (total_correct / total_trials) if total_trials else np.nan
        feats['A3_valid_rt'] = safe_mean(rt[valid_mask])
        feats['A3_invalid_rt'] = safe_mean(rt[invalid_mask])
    else:
        feats['A3_valid_accuracy'] = feats['A3_invalid_accuracy'] = feats['A3_total_accuracy'] = np.nan
        feats['A3_valid_rt'] = feats['A3_invalid_rt'] = np.nan
    if rt.size:
        correct_mask = responses == 1
        incorrect_mask = responses == 0
        feats['A3_correct_rt'] = safe_mean(rt[correct_mask]) if correct_mask.any() else np.nan
        feats['A3_incorrect_rt'] = safe_mean(rt[incorrect_mask]) if incorrect_mask.any() else np.nan
    else:
        feats['A3_correct_rt'] = feats['A3_incorrect_rt'] = np.nan
    feats['A3_accuracy_gap'] = safe_diff(feats.get('A3_valid_accuracy', np.nan), feats.get('A3_invalid_accuracy', np.nan))
    return feats

def compute_a4_features(row):
    condition = parse_int_sequence(row.get('A4_Condition'))
    response_raw = parse_int_sequence(row.get('A4_Response1'))
    rt = parse_float_sequence(row.get('A4_ResponseTime'))
    condition, response_raw, rt = align_sequences(condition, response_raw, rt)
    responses = np.where(response_raw == 1, 1.0, np.where(response_raw == 2, 0.0, np.nan)) if response_raw.size else np.array([])
    feats = {}
    if responses.size:
        feats['A4_response_rate'] = safe_rate(responses)
        congruent_mask = condition == 1
        incongruent_mask = condition == 2
        feats['A4_congruent_accuracy'] = safe_rate(responses[congruent_mask]) if congruent_mask.any() else np.nan
        feats['A4_incongruent_accuracy'] = safe_rate(responses[incongruent_mask]) if incongruent_mask.any() else np.nan
        feats['A4_mean_rt_con'] = safe_mean(rt[congruent_mask]) if congruent_mask.any() else np.nan
        feats['A4_mean_rt_incon'] = safe_mean(rt[incongruent_mask]) if incongruent_mask.any() else np.nan
    else:
        feats['A4_response_rate'] = np.nan
        feats['A4_congruent_accuracy'] = feats['A4_incongruent_accuracy'] = np.nan
        feats['A4_mean_rt_con'] = feats['A4_mean_rt_incon'] = np.nan
    feats['A4_accuracy_gap'] = safe_diff(feats.get('A4_incongruent_accuracy', np.nan), feats.get('A4_congruent_accuracy', np.nan))
    feats['A4_rt_gap'] = safe_diff(feats.get('A4_mean_rt_incon', np.nan), feats.get('A4_mean_rt_con', np.nan))
    return feats

def compute_a5_features(row):
    change_type = parse_int_sequence(row.get('A5_ChangeType'))
    responses = parse_int_sequence(row.get('A5_Response'))
    change_type, responses = align_sequences(change_type, responses)
    feats = {}
    def acc_for(code):
        mask = change_type == code
        return safe_rate(responses[mask]) if mask.any() else np.nan
    feats['A5_accuracy_non_change'] = acc_for(1)
    feats['A5_accuracy_pos_change'] = acc_for(2)
    feats['A5_accuracy_color_change'] = acc_for(3)
    feats['A5_accuracy_shape_change'] = acc_for(4)
    acc_values = np.array([
        feats['A5_accuracy_non_change'],
        feats['A5_accuracy_pos_change'],
        feats['A5_accuracy_color_change'],
        feats['A5_accuracy_shape_change']
    ], dtype=float)
    valid_acc = acc_values[~np.isnan(acc_values)]
    feats['A5_accuracy_var'] = float(np.nanvar(valid_acc)) if valid_acc.size > 1 else np.nan
    return feats

def compute_a6_a7_features(row, stats=None):
    a6_score = pd.to_numeric(row.get('A6_Count'), errors='coerce')
    a7_score = pd.to_numeric(row.get('A7_Count'), errors='coerce')
    def zscore(value, mean, std):
        if pd.isna(value) or pd.isna(std) or std == 0:
            return np.nan
        return float((value - mean) / std)
    
    if stats is None:
        stats = {}
    
    return {
        'A6_score': a6_score,
        'A6_zscore': zscore(a6_score, stats.get('A6_mean'), stats.get('A6_std')),
        'A7_score': a7_score,
        'A7_zscore': zscore(a7_score, stats.get('A7_mean'), stats.get('A7_std'))
    }

def compute_a8_features(row, stats=None):
    distortion = pd.to_numeric(row.get('A8_Count1'), errors='coerce')
    consistency = pd.to_numeric(row.get('A8_Count2'), errors='coerce')
    
    if stats is None:
        stats = {}
    
    threshold = stats.get('A8_distortion_threshold')
    if pd.isna(distortion) or pd.isna(threshold):
        flag = np.nan
    else:
        flag = int(distortion >= threshold)
    return {
        'A8_distortion_score': distortion,
        'A8_consistency_score': consistency,
        'A8_distortion_flag': flag
    }

def compute_a9_features(row):
    counts = [pd.to_numeric(row.get(f'A9_Count{i}'), errors='coerce') for i in range(1, 6)]
    emotional, behavior, reality, cognitive, stress = counts
    total_score = float(np.nanmean(counts)) if any(not pd.isna(c) for c in counts) else np.nan
    stability_gap = emotional - behavior if not (pd.isna(emotional) or pd.isna(behavior)) else np.nan
    return {
        'A9_emotional_stability': emotional,
        'A9_behavior_stability': behavior,
        'A9_reality_checking': reality,
        'A9_cognitive_agility': cognitive,
        'A9_stress_level': stress,
        'A9_total_score': total_score,
        'A9_stability_gap': stability_gap
    }

def compute_a_features(row, stats=None):
    feats = {
        'Test_id': row['Test_id'],
        'Test': row.get('Test', 'A')
    }
    if 'Label' in row.index:
        feats['Label'] = row['Label']
    feats.update(compute_a1_features(row))
    feats.update(compute_a2_features(row))
    feats.update(compute_a3_features(row))
    feats.update(compute_a4_features(row))
    feats.update(compute_a5_features(row))
    feats.update(compute_a6_a7_features(row, stats))
    feats.update(compute_a8_features(row, stats))
    feats.update(compute_a9_features(row))
    return feats

def build_a_features(df, stats=None, label_df=None):
    """Hybrid (Vectorized + Rename) A-feature builder.

    - Step 1: Apply explicit RENAME_MAP_A for clarity and stability
    - Step 2: Vectorized feature extraction
        * Sequence-like columns (comma-separated strings): mean/std/min/max/median/range
        * Scalar numeric columns: raw value + column-wise z-score (normalized)

    Notes:
    - Avoids row-wise iterloops for performance on large datasets
    - Keeps signature compatible with existing pipeline
    """
    import time
    start_t = time.time()

    # 1) Rename columns explicitly
    renamed = rename_a_columns(df.copy())
    if label_df is not None and 'Label' not in renamed.columns:
        # Optional: attach labels when provided
        if 'Test_id' in renamed.columns and 'Test_id' in label_df.columns:
            renamed = renamed.merge(label_df[['Test_id', 'Label']], on='Test_id', how='left')

    # 2) Initialize features frame
    features = pd.DataFrame()
    if 'Test_id' in renamed.columns:
        features['Test_id'] = renamed['Test_id']

    # 3) Vectorized scan over columns
    new_cols_dict = {}
    seq_stats_per_col = 0
    scalar_cols = 0

    # Columns to skip outright
    skip_cols = {'Test_id', 'Test', 'Label', 'PrimaryKey', 'TestDate'}

    for col in renamed.columns:
        if col in skip_cols:
            continue
        col_series = renamed[col]

        # Heuristic: treat as sequence if first non-null contains a comma
        first_valid = col_series.dropna().iloc[0] if col_series.notna().any() else None
        if isinstance(first_valid, str) and (',' in first_valid):
            # Sequence-like column -> parse once per row (vectorized over series)
            seqs = col_series.astype(str).apply(lambda s: parse_sequence(s) if (',' in s) else np.array([], dtype=np.float64))
            new_cols_dict[f'{col}_mean'] = [np.nan if arr.size == 0 else float(np.nanmean(arr)) for arr in seqs]
            new_cols_dict[f'{col}_std'] = [np.nan if arr.size == 0 else float(np.nanstd(arr)) for arr in seqs]
            new_cols_dict[f'{col}_min'] = [np.nan if arr.size == 0 else float(np.nanmin(arr)) for arr in seqs]
            new_cols_dict[f'{col}_max'] = [np.nan if arr.size == 0 else float(np.nanmax(arr)) for arr in seqs]
            new_cols_dict[f'{col}_median'] = [np.nan if arr.size == 0 else float(np.nanmedian(arr)) for arr in seqs]
            new_cols_dict[f'{col}_range'] = [np.nan if arr.size == 0 else (float(np.nanmax(arr) - np.nanmin(arr)) if arr.size > 1 else 0.0) for arr in seqs]
            seq_stats_per_col += 1
        else:
            # Try scalar numeric column
            numeric_series = pd.to_numeric(col_series, errors='coerce')
            if numeric_series.notna().any():
                new_cols_dict[f'{col}_value'] = numeric_series.values
                mean_val = float(numeric_series.mean()) if numeric_series.notna().any() else np.nan
                std_val = float(numeric_series.std()) if numeric_series.notna().any() else np.nan
                if pd.notna(std_val) and std_val != 0:
                    new_cols_dict[f'{col}_normalized'] = ((numeric_series - mean_val) / std_val).values
                scalar_cols += 1
            # else: non-numeric and non-sequence -> skip

    if new_cols_dict:
        features = pd.concat([features, pd.DataFrame(new_cols_dict)], axis=1)

    elapsed = time.time() - start_t
    print(f"A features built (hybrid): {features.shape[1] - (1 if 'Test_id' in features.columns else 0)} cols | seq_cols={seq_stats_per_col}, scalar_cols={scalar_cols}, time={elapsed:.1f}s")
    return features

def compute_b1_features(row):
    seq1 = parse_int_sequence(row.get('B1_Response1'))
    seq2 = parse_float_sequence(row.get('B1_ResponseTime'))
    seq3 = parse_int_sequence(row.get('B1_Response2'))
    seq1, seq2, seq3 = align_sequences(seq1, seq2, seq3)
    feats = {
        'B1_task1_accuracy': binary_accuracy(seq1)
    }
    change_correct = (seq3 == 1).sum()
    change_incorrect = (seq3 == 2).sum()
    change_total = change_correct + change_incorrect
    feats['B1_task2_change_acc'] = (change_correct / change_total) if change_total else np.nan
    non_correct = (seq3 == 3).sum()
    non_incorrect = (seq3 == 4).sum()
    non_total = non_correct + non_incorrect
    feats['B1_task2_non_change_acc'] = (non_correct / non_total) if non_total else np.nan
    feats['B1_task2_accuracy_gap'] = safe_diff(feats['B1_task2_change_acc'], feats['B1_task2_non_change_acc'])
    feats['B1_task2_mean_rt'] = mean_ignore_zeros(seq2)
    return feats

def compute_b2_features(row):
    seq1 = parse_int_sequence(row.get('B2_Response1'))
    seq2 = parse_float_sequence(row.get('B2_ResponseTime'))
    seq3 = parse_int_sequence(row.get('B2_Response2'))
    seq1, seq2, seq3 = align_sequences(seq1, seq2, seq3)
    feats = {
        'B2_task1_accuracy': binary_accuracy(seq1)
    }
    change_correct = (seq3 == 1).sum()
    change_incorrect = (seq3 == 2).sum()
    change_total = change_correct + change_incorrect
    feats['B2_task2_change_acc'] = (change_correct / change_total) if change_total else np.nan
    non_correct = (seq3 == 3).sum()
    non_incorrect = (seq3 == 4).sum()
    non_total = non_correct + non_incorrect
    feats['B2_task2_non_change_acc'] = (non_correct / non_total) if non_total else np.nan
    feats['B2_task2_accuracy_gap'] = safe_diff(feats['B2_task2_change_acc'], feats['B2_task2_non_change_acc'])
    feats['B2_task2_mean_rt'] = mean_ignore_zeros(seq2)
    return feats

def compute_b3_features(row):
    responses = parse_int_sequence(row.get('B3_Response'))
    rt = parse_float_sequence(row.get('B3_ResponseTime'))
    responses, rt = align_sequences(responses, rt)
    return {
        'B3_accuracy': binary_accuracy(responses),
        'B3_mean_rt': mean_ignore_zeros(rt)
    }

def compute_b4_features(row):
    responses = parse_int_sequence(row.get('B4_Response'))
    rt = parse_float_sequence(row.get('B4_ResponseTime'))
    responses, rt = align_sequences(responses, rt)
    congruent_mask = np.isin(responses, [1, 2])
    incongruent_mask = np.isin(responses, [3, 4])
    feats = {
        'B4_congruent_accuracy': binary_accuracy(responses[congruent_mask]) if congruent_mask.any() else np.nan,
        'B4_incongruent_accuracy': binary_accuracy(responses[incongruent_mask]) if incongruent_mask.any() else np.nan,
        'B4_mean_rt_congruent': mean_ignore_zeros(rt[congruent_mask]) if congruent_mask.any() else np.nan,
        'B4_mean_rt_incongruent': mean_ignore_zeros(rt[incongruent_mask]) if incongruent_mask.any() else np.nan
    }
    feats['B4_accuracy_gap'] = safe_diff(feats['B4_incongruent_accuracy'], feats['B4_congruent_accuracy'])
    feats['B4_rt_gap'] = safe_diff(feats['B4_mean_rt_incongruent'], feats['B4_mean_rt_congruent'])
    return feats

def compute_b5_features(row):
    responses = parse_int_sequence(row.get('B5_Response'))
    rt = parse_float_sequence(row.get('B5_ResponseTime'))
    responses, rt = align_sequences(responses, rt)
    return {
        'B5_accuracy': binary_accuracy(responses),
        'B5_mean_rt': mean_ignore_zeros(rt)
    }

def compute_b6_features(row):
    responses = parse_int_sequence(row.get('B6_Response'))
    return {
        'B6_accuracy': binary_accuracy(responses)
    }

def compute_b7_features(row):
    responses = parse_int_sequence(row.get('B7_Response'))
    return {
        'B7_accuracy': binary_accuracy(responses)
    }

def compute_b8_features(row):
    responses = parse_int_sequence(row.get('B8_Response'))
    return {
        'B8_accuracy': binary_accuracy(responses)
    }

def compute_b9_features(row):
    return {
        'B9_aud_hit': pd.to_numeric(row.get('B9_Count1'), errors='coerce'),
        'B9_aud_miss': pd.to_numeric(row.get('B9_Count2'), errors='coerce'),
        'B9_aud_fa': pd.to_numeric(row.get('B9_Count3'), errors='coerce'),
        'B9_aud_cr': pd.to_numeric(row.get('B9_Count4'), errors='coerce'),
        'B9_vis_err': pd.to_numeric(row.get('B9_Count5'), errors='coerce')
    }

def compute_b10_features(row):
    return {
        'B10_aud_hit': pd.to_numeric(row.get('B10_Count1'), errors='coerce'),
        'B10_aud_miss': pd.to_numeric(row.get('B10_Count2'), errors='coerce'),
        'B10_aud_fa': pd.to_numeric(row.get('B10_Count3'), errors='coerce'),
        'B10_aud_cr': pd.to_numeric(row.get('B10_Count4'), errors='coerce'),
        'B10_vis1_err': pd.to_numeric(row.get('B10_Count5'), errors='coerce'),
        'B10_vis2_correct': pd.to_numeric(row.get('B10_Count6'), errors='coerce')
    }

def compute_b_features(row):
    feats = {
        'Test_id': row['Test_id'],
        'Test': row.get('Test', 'B')
    }
    if 'Label' in row.index:
        feats['Label'] = row['Label']
    feats.update(compute_b1_features(row))
    feats.update(compute_b2_features(row))
    feats.update(compute_b3_features(row))
    feats.update(compute_b4_features(row))
    feats.update(compute_b5_features(row))
    feats.update(compute_b6_features(row))
    feats.update(compute_b7_features(row))
    feats.update(compute_b8_features(row))
    feats.update(compute_b9_features(row))
    feats.update(compute_b10_features(row))
    return feats

def build_b_features(df, label_df=None):
    """Hybrid (Vectorized + Rename) B-feature builder.

    Applies RENAME_MAP_B, then extracts features vectorized:
    - Sequence-like columns: mean/std/min/max/median/range
    - Scalar numeric columns: value + z-score
    """
    import time
    start_t = time.time()

    # 1) Rename
    renamed = rename_b_columns(df.copy())
    if label_df is not None and 'Label' not in renamed.columns:
        if 'Test_id' in renamed.columns and 'Test_id' in label_df.columns:
            renamed = renamed.merge(label_df[['Test_id', 'Label']], on='Test_id', how='left')

    # 2) Initialize features
    features = pd.DataFrame()
    if 'Test_id' in renamed.columns:
        features['Test_id'] = renamed['Test_id']

    # 3) Vectorized extraction
    new_cols_dict = {}
    seq_stats_per_col = 0
    scalar_cols = 0
    skip_cols = {'Test_id', 'Test', 'Label', 'PrimaryKey', 'TestDate'}

    for col in renamed.columns:
        if col in skip_cols:
            continue
        col_series = renamed[col]
        first_valid = col_series.dropna().iloc[0] if col_series.notna().any() else None
        if isinstance(first_valid, str) and (',' in first_valid):
            seqs = col_series.astype(str).apply(lambda s: parse_sequence(s) if (',' in s) else np.array([], dtype=np.float64))
            new_cols_dict[f'{col}_mean'] = [np.nan if arr.size == 0 else float(np.nanmean(arr)) for arr in seqs]
            new_cols_dict[f'{col}_std'] = [np.nan if arr.size == 0 else float(np.nanstd(arr)) for arr in seqs]
            new_cols_dict[f'{col}_min'] = [np.nan if arr.size == 0 else float(np.nanmin(arr)) for arr in seqs]
            new_cols_dict[f'{col}_max'] = [np.nan if arr.size == 0 else float(np.nanmax(arr)) for arr in seqs]
            new_cols_dict[f'{col}_median'] = [np.nan if arr.size == 0 else float(np.nanmedian(arr)) for arr in seqs]
            new_cols_dict[f'{col}_range'] = [np.nan if arr.size == 0 else (float(np.nanmax(arr) - np.nanmin(arr)) if arr.size > 1 else 0.0) for arr in seqs]
            seq_stats_per_col += 1
        else:
            numeric_series = pd.to_numeric(col_series, errors='coerce')
            if numeric_series.notna().any():
                new_cols_dict[f'{col}_value'] = numeric_series.values
                mean_val = float(numeric_series.mean()) if numeric_series.notna().any() else np.nan
                std_val = float(numeric_series.std()) if numeric_series.notna().any() else np.nan
                if pd.notna(std_val) and std_val != 0:
                    new_cols_dict[f'{col}_normalized'] = ((numeric_series - mean_val) / std_val).values
                scalar_cols += 1

    if new_cols_dict:
        features = pd.concat([features, pd.DataFrame(new_cols_dict)], axis=1)

    elapsed = time.time() - start_t
    print(f"B features built (hybrid): {features.shape[1] - (1 if 'Test_id' in features.columns else 0)} cols | seq_cols={seq_stats_per_col}, scalar_cols={scalar_cols}, time={elapsed:.1f}s")
    return features
