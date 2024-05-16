import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import pandas as pd


def plot_psd(signal):
    plt.figure(figsize=(10, 6))
    plt.psd(signal, NFFT=2048, Fs=1/60, window=np.hamming(2048))
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.show()


def specific_frequency_filter(time_series, sampling_rate, frequency, bandwidth):
    # Compute the Fourier transform
    fft_vals = np.fft.fft(time_series)
    frequencies = np.fft.fftfreq(time_series.size, d=1 / sampling_rate)
    fft_vals[np.abs(frequencies) < (frequency - bandwidth / 2)] = 0
    fft_vals[np.abs(frequencies) > (frequency + bandwidth / 2)] = 0
    filtered_time_series = np.fft.ifft(fft_vals)
    return np.real(filtered_time_series)


def lh_pass_filter(time_series, sampling_rate, lower_bound, upper_bound):
    fft_vals = np.fft.fft(time_series)
    frequencies = np.fft.fftfreq(time_series.size, d=1 / sampling_rate)
    fft_vals[np.abs(frequencies) < lower_bound] = 0
    fft_vals[np.abs(frequencies) > upper_bound] = 0
    filtered_time_series = np.fft.ifft(fft_vals)
    return np.real(filtered_time_series)


def reorder_states_by_means(model, hidden_states):
    means = model.means_.flatten()
    order = np.argsort(means)
    model.means_ = model.means_[order]
    model.covars_ = model.covars_[order]
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[order, :][:, order]
    hidden_states = np.array([np.where(order == state)[0][0] for state in hidden_states])
    return hidden_states, model


def fourier_based_hmm(signal, nr_comp=2, lowpass=0.00001, highpass=0.0002):
    filtered = lh_pass_filter(signal, 1/60, lowpass, highpass)
    model = hmm.GaussianHMM(n_components=nr_comp, covariance_type="full", n_iter=100, random_state=123456)
    signal_reshaped = np.expand_dims(signal, axis=1)
    model.fit(signal_reshaped)
    hidden_states = model.predict(signal_reshaped)
    hidden_states, model = reorder_states_by_means(model, hidden_states)
    return pd.Series(hidden_states, index=signal.index)

