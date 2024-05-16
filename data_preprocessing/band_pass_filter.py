import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', output="ba")
    return b, a


def apply_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data)
    return pd.Series(y, index=data.index)


def get_vector_magnitude(ax, ay, az):
    return np.sqrt(ax**2+ay**2+az**2)


def set_time_index(activity_df: pd.DataFrame):
    activity_df['time'] = pd.to_datetime(activity_df['time'])
    activity_df.set_index('time', inplace=True)
    return activity_df


def accumulate_signal(activity_signal, epoch_length):
    # Rectify the signal
    rectified_signal = activity_signal.abs()
    if epoch_length:
        rectified_signal = rectified_signal.resample(f'{epoch_length}S').sum()
    # Accumulate signal
    return rectified_signal


def preprocess_signal(activity_signal: pd.Series, epoch_length, lowcut, highcut, order):
    filtered_signal = apply_bandpass_filter(activity_signal, lowcut, highcut, 100, order)
    accumulated_signal = accumulate_signal(filtered_signal, epoch_length)
    return accumulated_signal


def preprocessing_from_dataframe(activity_file:  pd.DataFrame, epoch_length, lowcut=0.25, highcut=2.5, order=3):
    signal_axis = {}
    for axis in ["x", "y", "z"]:
        signal_dimension = activity_file[axis]
        processed_signal = preprocess_signal(signal_dimension, epoch_length, lowcut, highcut, order)
        signal_axis[axis] = processed_signal
    return pd.DataFrame(signal_axis)
