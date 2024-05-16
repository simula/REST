import pandas as pd
import numpy as np
from scipy.fft import fft
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import iqr, skew, kurtosis


def mean_abs_diff(data):
    return np.mean(np.abs(np.diff(data)))


def autocorrelation(data, lag=1):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    data0 = data[:-lag] - mean
    data1 = data[lag:] - mean
    return np.sum(data0 * data1) / (n * var)


def add_zeros_end(series, n_zeros):
    # Get the last datetime in the series and add one minute for each zero you want to add
    last_datetime = series.index[-1]
    new_datetimes = pd.date_range(start=last_datetime + timedelta(minutes=1), periods=n_zeros, freq='T')

    # Create a new series of zeros
    zeros_series = pd.Series([0]*n_zeros, index=new_datetimes)

    # Concatenate the original series with the new series of zeros
    new_series = pd.concat([series, zeros_series])

    return new_series


def add_zeros_beginning(series, n_zeros):
    # Get the first datetime in the series and subtract one minute for each zero you want to add
    first_datetime = series.index[0]
    new_datetimes = pd.date_range(end=first_datetime - timedelta(minutes=1), periods=n_zeros, freq='T')
    zeros_series = pd.Series([0]*n_zeros, index=new_datetimes)
    new_series = pd.concat([zeros_series, series])
    return new_series


def power_spectrum(window, sampling_rate):
    n = len(window)
    window_array = window.values  # Convert to numpy array
    fft_values = fft(window_array)
    psd_values = np.abs(fft_values) ** 2
    fft_freq = np.fft.fftfreq(n, d=1 / sampling_rate)[:n // 2]
    psd_values = 2 * psd_values[:n // 2]
    return fft_freq, psd_values


def calculate_power_spectrum(signal, window_length, sampling_rate):
    power_spectrums = []
    # Calculate power spectrum for each sliding window
    signal = add_zeros_end(signal, window_length)
    signal = add_zeros_beginning(signal, window_length)
    for i in range(window_length, len(signal) - window_length):
        window = signal[i:i+window_length]
        fft_freq, psd_values = power_spectrum(window, sampling_rate)
        power_spectrums.append(max(psd_values))
    return pd.Series(power_spectrums, name=signal.name)


def spectrum_non_wear_times(signals_file, epoch, window_length=10, threshold=2*10^-5):
    psd_x = calculate_power_spectrum(signals_file["x"], window_length, 1/epoch).reset_index(drop=True)
    psd_y = calculate_power_spectrum(signals_file["y"], window_length, 1/epoch).reset_index(drop=True)
    psd_z = calculate_power_spectrum(signals_file["z"], window_length, 1/epoch).reset_index(drop=True)
    psd_df = pd.concat([psd_x, psd_y, psd_z], axis=1)
    max_psd = psd_df.max(axis=1)
    non_wear_signal = (max_psd >= threshold).astype(int)
    non_wear_signal.index = signals_file.index
    return non_wear_signal


def calculate_features(window_ts):
    return {"std": np.std(window_ts),
            "mean": np.mean(window_ts),
            "median": np.median(window_ts),
            "mean_abs_diff": mean_abs_diff(window_ts),
            "autocorrelation": autocorrelation(window_ts),
            "iqr": iqr(window_ts),
            "mad": np.mean(np.abs(window_ts - np.mean(window_ts))),
            "skewness": skew(window_ts),
            "kurtosis": kurtosis(window_ts)}


def calculate_window_features(signal, window_length):
    features_windows = []
    # Calculate power spectrum for each sliding window
    #signal = add_zeros_end(signal, window_length)
    #signal = add_zeros_beginning(signal, window_length)
    for i in range(0, len(signal) - window_length):
        features_windows.append(calculate_features(signal[i:i + window_length]))
    return pd.DataFrame(features_windows)


def window_clustering(features_df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    pca= PCA(n_components=2).fit(scaled_features)
    principal_components = pca.transform(scaled_features)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    clusters = KMeans(n_clusters=2, random_state=0, n_init="auto").fit_predict(principal_components)
    pca_df['Cluster'] = clusters
    return pca_df


def map_clusters_to_timeseries(pca_df, window_length, n_samples):
    # Initialize an empty array for the cluster labels
    cluster_labels = np.zeros(n_samples)

    # Iterate over each window
    for i in range(len(pca_df)):
        # Assign the cluster label of the window to all data points in the window
        cluster_labels[i:i + window_length] = pca_df.loc[i, 'Cluster']

    return cluster_labels


def plot_clusters(df):
    fig, ax = plt.subplots()

    # List of markers can be extended depending on the number of clusters
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for cluster in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster]
        ax.scatter(cluster_df['PC1'], cluster_df['PC2'], marker=markers[cluster % len(markers)], label=f'Cluster {cluster}')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    plt.show()


def detect_and_analyze_periods(time_series, threshold, min_length):
    # Initialize the list of detected periods
    detected_periods_start = []
    detected_periods_end = []
    # Initialize the start and end indices of the current period
    start = None
    end = None

    # Get the datetime index
    datetime_index = time_series.index

    # Iterate over the time series
    for i, value in enumerate(time_series):
        # If the current value is below the threshold and we are not currently in a period
        if value < threshold and start is None:
            # Start a new period
            start = i
        # If the current value is above the threshold and we are currently in a period
        elif value >= threshold and start is not None:
            # End the current period
            end = i
            # If the length of the period is at least min_length
            if end - start >= min_length:
                detected_periods_start.append(datetime_index[start])
                detected_periods_end.append(datetime_index[end])
            # Reset the start and end indices
            start = None
            end = None

    # If the last period goes until the end of the time series
    if start is not None and end is None:
        end = len(time_series)
        # If the length of the period is at least min_length
        if end - start >= min_length:
            detected_periods_start.append(datetime_index[start])
            detected_periods_end.append(datetime_index[end-1])

    feature_df = pd.DataFrame()
    feature_df["start"] = detected_periods_start
    feature_df["end"] = detected_periods_end
    return feature_df


def get_wear_non_wear_series(time_series, time_df):
    # Initialize a new Series with the same index as the time series, and set all values to "w"
    wear_non_wear_series = pd.Series(1, index=time_series.index)
    # Iterate over the rows of the DataFrame
    for index, row in time_df.iterrows():
        # Get the start and end times of the current row
        start_time = row["start"]
        end_time = row["end"]
        # Set the corresponding range in the wear/non-wear Series to "n"
        wear_non_wear_series.loc[start_time:end_time] = 0
    return wear_non_wear_series


def threshold_non_wear_times(signal, min_length, threshold):
    times = detect_and_analyze_periods(signal, threshold, min_length)
    return get_wear_non_wear_series(signal, times)



