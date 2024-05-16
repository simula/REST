import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rule_a(series):
    return series.where(~(series.shift(-4).rolling(5).sum() == 1) | (series == 0), 1)


def rule_b(series):
    return series.where(~(series.shift(-10).rolling(11).sum() == 0) | (series == 1), 0)


def rule_c(series):
    return series.where(~(series.shift(-15).rolling(16).sum() == 0) | (series == 1), 0)


def rule_d(series):
    rolling_sum_before = series.shift(-10).rolling(11).sum()
    rolling_sum_after = series.shift(10).rolling(11).sum()
    return series.where(~((series.rolling(11, center=True).sum() <= 6) & ((rolling_sum_before == 0) | (rolling_sum_after == 0))), 0)


def rule_e(series):
    rolling_sum_before = series.shift(-20).rolling(21).sum()
    rolling_sum_after = series.shift(20).rolling(21).sum()
    return series.where(~((series.rolling(21, center=True).sum() <= 10) & ((rolling_sum_before == 0) | (rolling_sum_after == 0))), 0)


def calculate_total_vector(signals_df: pd.DataFrame) -> pd.Series:
    signals_df = signals_df[["x", "y", "z"]]
    return np.sqrt((signals_df**2).sum(axis=1))


def pad_signal(signal, padding_length, mode):
    if mode == 'start':
        return pd.concat([pd.Series([0] * padding_length), signal])
    elif mode == 'end':
        return pd.concat([signal, pd.Series([0] * padding_length)])
    elif mode == 'both':
        return pd.concat([pd.Series([0] * padding_length), signal, pd.Series([0] * padding_length)])
    else:
        raise ValueError("Mode should be 'start', 'end', or 'both'")


def nat_function(x):
    return len([i for i in x if i > 0])


def sadeh_sleep_stage_algo_(signal):
    if not isinstance(signal, pd.Series):
        raise TypeError('Input signal should be a pandas Series')
    signal_ = signal.clip(upper=300)
    mean_signal = pad_signal(signal, 5, 'both').rolling(11, center=True).mean()[5:-5].reset_index(drop=True)
    std_signal = pad_signal(signal_, 5, 'start').rolling(6).std()[5:].reset_index(drop=True)
    nat_signal = pad_signal(signal_, 5, 'both').rolling(11, center=True).apply(nat_function, raw=False)[5:-5].reset_index(drop=True)
    log_signal = signal_.apply(lambda x: np.log(x + 1)).reset_index(drop=True)
    sadeh_signal = 7.601 - (0.065 * mean_signal) - (1.08 * nat_signal) - (0.056 * std_signal) - (0.703 * log_signal)
    sleep_stages = (sadeh_signal < sadeh_signal.describe()["50%"]).astype(int).values
    return pd.Series(sleep_stages, index=signal.index)

def calculate_cole_kripke_60s(signal):
    # Standard Cole-Kripke weights for 1-minute epochs
    weights = [1.08, 0.92, 0.82, 1, 0.82, 0.92, 1.08]
    # Normalize activity counts if necessary, typically data are scaled to a maximum of 100 or 1
    signal = signal / 100
    # Apply clipping based on standard actigraphy range if necessary
    signal = signal.clip(upper=3)  # Change according to your data specifics
    # Apply convolution with the predefined weights
    weighted_activity = np.convolve(signal.values, weights, mode='same')  # 'same' keeps original length
    # Standard threshold for sleep-wake decision is typically around 1 but can vary
    sleep_wake = (weighted_activity < 1).astype(int)
    return pd.Series(sleep_wake, index=signal.index)

def sadeh_sleep_stage_algo(signal):
    # Clip signal to a typical maximum for actigraphy; adjust as necessary
    signal = signal.clip(upper=300)
    # Calculate rolling statistics based on original Sadeh specifications
    mean_signal = signal.rolling(window=11, center=True).mean()
    std_signal = signal.rolling(window=11, center=True).std()
    # Natural activity threshold in original Sadeh was typically based on counts
    nat_signal = signal.apply(lambda x: x > 50).rolling(window=11, center=True).sum()
    log_signal = np.log(signal + 1)
    # Calculate the Sadeh score and determine sleep-wake status
    sadeh_score = 7.601 - 0.065 * mean_signal - 1.08 * nat_signal - 0.056 * std_signal - 0.703 * log_signal
    sleep_wake = (sadeh_score < 0).astype(int)
    return pd.Series(sleep_wake, index=signal.index)

def calculate_oakley(activity_counts, threshold=40):
    # Default Oakley algorithm parameters for 1-minute epochs
    n = len(activity_counts)
    results = []

    for i in range(n):
        current_count = activity_counts[i]

        # Awake if current activity exceeds threshold
        if current_count > threshold:
            results.append(1)  # Awake
            continue

        # Sum activity in surrounding epochs; adjust these indices as necessary for different contexts
        start_index = max(0, i - 1)
        end_index = min(n, i + 2)  # Note: end_index is exclusive

        surrounding_sum = sum(activity_counts[start_index:end_index])

        # Awake if surrounding activity exceeds a set sum, else asleep
        if surrounding_sum > 100:  # Adjust total activity threshold as needed
            results.append(1)  # Awake
        else:
            results.append(0)  # Sleep

    return pd.Series(results, index=activity_counts.index)

def calculate_cole_kripke_(signal):
    weights = [0.001*106, 0.001*54, 0.001*58, 0.001*76, 0.001*230, 0.001*74, 0.001*67]
    weights.reverse()
    signal = signal/100
    signal = signal.clip(upper=300)
    weighted_activity = np.convolve(signal.values, weights, mode='full')
    # Truncate the convolved signal to the size of the input signal
    weighted_activity = weighted_activity[:len(signal)]
    sleep_wake = (weighted_activity > 1).astype(int)
    return pd.Series(sleep_wake, index=signal.index)


def sazonova_algorithm(activity_counts, activity_threshold=40, window_size=11, wake_threshold=2):
    """
    Apply the Sazonova sleep-wake algorithm.

    Parameters:
    - activity_counts (pd.Series): The activity counts data.
    - activity_threshold (int): Threshold for an epoch to be considered active.
    - window_size (int): Size of the moving window.
    - wake_threshold (int): Number of active epochs within the window needed to classify the middle epoch as wake.

    Returns:
    pd.Series: Binary series where 1 indicates wake and 0 indicates sleep.
    """

    # Classify epochs as active/inactive
    is_active = activity_counts > activity_threshold

    # Initialize results with 0 (sleep)
    results = pd.Series(0, index=activity_counts.index)

    half_window = window_size // 2

    for i in range(half_window, len(activity_counts) - half_window):
        window_data = is_active.iloc[i - half_window: i + half_window + 1]
        if window_data.sum() > wake_threshold:
            results.iloc[i] = 1  # Wake
    return pd.Series(results, index=activity_counts.index)

def calculate_cole_kripke_30s(signal):
    # Cole-Kripke algorithm originally designed for 1-minute epochs
    # Adaptation for 30-second epochs may involve adjusting the weighting and threshold
    # Original weights based on minute-long epochs might need to be recalibrated
    weights = [0.5 * 1.08, 0.5 * 0.92, 0.5 * 0.82, 1.0 * 1, 0.5 * 0.82, 0.5 * 0.92, 0.5 * 1.08]
    signal = signal / 100  # Normalize activity counts if necessary
    signal = signal.clip(upper=3)  # Adjust based on typical maximum for 30-second epochs
    weighted_activity = np.convolve(signal.values, weights, mode='same')  # 'same' keeps original length
    sleep_wake = (weighted_activity < 1).astype(int)  # Adjust threshold based on your dataset and validation
    return pd.Series(sleep_wake, index=signal.index)

def sadeh_sleep_stage_algo_30s(signal):
    signal = signal.clip(upper=300)  # Adjust clipping based on your data's range
    mean_signal = signal.rolling(window=11, center=True).mean()
    std_signal = signal.rolling(window=11, center=True).std()
    nat_signal = signal.apply(lambda x: x > 50).rolling(window=11, center=True).sum()  # Active count threshold might need adjustment
    log_signal = np.log(signal + 1)
    sadeh_score = 7.601 - 0.065 * mean_signal - 1.08 * nat_signal - 0.056 * std_signal - 0.703 * log_signal
    sleep_wake = (sadeh_score < 0).astype(int)  # Adjust threshold based on validation with your data
    return pd.Series(sleep_wake, index=signal.index)

def calculate_oakley_30s(activity_counts, threshold=40, surrounding_minutes=2):
    # For 30-second epochs, surrounding_minutes might be adjusted
    # This algorithm usually counts minutes, so for 30-sec epochs, consider the appropriate conversion
    results = []
    for i in range(len(activity_counts)):
        window_start = max(0, i - surrounding_minutes * 2)  # Adjust for 30-sec epochs
        window_end = min(len(activity_counts), i + surrounding_minutes * 2 + 1)
        window_sum = activity_counts[window_start:window_end].sum()
        if activity_counts[i] > threshold or window_sum > 100:  # Adjust thresholds based on your data
            results.append(1)  # Awake
        else:
            results.append(0)  # Sleep
    return pd.Series(results, index=activity_counts.index)

def calculate_oakley_(activity_counts, T=40, S=100, N=2):
    n = len(activity_counts)
    results = []

    for i in range(n):
        current_count = activity_counts[i]

        if current_count > T:
            results.append(1)  # Awake
            continue

        # Define the start and end indices to sum over the surrounding N minutes
        start_index = max(0, i - N)
        end_index = min(n, i + N + 1)  # +1 because Python slices are exclusive at the end

        surrounding_sum = sum(activity_counts[start_index:end_index])

        if surrounding_sum > S:
            results.append(1)  # Awake
        else:
            results.append(0)  # Sleep

    return pd.Series(results, index=activity_counts.index)


def merge_non_wear_sleep_stages(non_wear_signal: pd.Series, sleep_stages: pd.Series):
    merged_series = non_wear_signal.copy()
    merged_series[merged_series == 0] = 'n'
    sleep_stages = sleep_stages.reindex(merged_series.index)
    mask = merged_series == 1
    merged_series[mask] = sleep_stages[mask].map({0: 's', 1: 'w'})
    return merged_series


def merge_non_wear_sleep_stages_binary(non_wear_signal: pd.Series, sleep_stages: pd.Series):
    merged_series = non_wear_signal.copy()
    sleep_stages[merged_series == 0] = 0
    return sleep_stages