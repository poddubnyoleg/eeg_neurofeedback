# -*- coding: utf-8 -*-
from scipy.signal import butter, lfilter, iirnotch, lfilter_zi
import pandas as pd
import numpy as np


class OnlineFilter:

    def __init__(self, fs, notch_f0, notch_q, low_cut, high_cut, order):
        self.zi_notch = ''
        self.zi_butter = ''

        self.notch_b, self.notch_a = iirnotch(notch_f0 / (fs / 2.), notch_q)
        self.butter_b, self.butter_a = butter(order, [low_cut/(0.5 * fs), high_cut/(0.5 * fs)], btype='band')

    def filter(self, data):

        if type(self.zi_butter) != np.ndarray:
            y, self.zi_notch = lfilter(self.notch_b, self.notch_a, data,
                                       zi=lfilter_zi(self.notch_b, self.notch_a)*data[0])
            y, self.zi_butter = lfilter(self.butter_b, self.butter_a, y,
                                        zi=lfilter_zi(self.butter_b, self.butter_a) * y[0])

        else:
            y, self.zi_notch = lfilter(self.notch_b, self.notch_a, data, zi=self.zi_notch)
            y, self.zi_butter = lfilter(self.butter_b, self.butter_a, y, zi=self.zi_butter)

        return y


# todo make windowed features instead of averaging of forecast
def spectral_features(x):
    # parameters taken following research
    return pd.Series(20 * np.log10(np.abs(np.fft.rfft(x*np.hanning(len(x)), n=256*2))[:80]))


class SpectralFeaturizer():

    def __init__(self, channels_number, window_seconds_size, sampling_rate, higher_freq_bound):

        self.channels_number = channels_number
        self.window_seconds_size = window_seconds_size
        self.sampling_rate = sampling_rate
        self.higher_freq_bound = higher_freq_bound
        self.buffer_size = self.sampling_rate*self.window_seconds_size

        self.data_buffer = np.zeros((self.buffer_size, self.channels_number))

    def calculate_features(self, new_data):

        self.data_buffer = np.append(self.data_buffer, new_data[:, :self.channels_number], axis=0)[ -self.buffer_size:, :]
        x = self.data_buffer
        # channels, seconds

        r = pd.DataFrame([pd.Series(10 * np.log10(np.abs(np.fft.rfft(x[:, i]*np.hanning(len(x[:, i])),
                                                                     n=self.sampling_rate*2))[
                                       :2*self.higher_freq_bound]))
                          for i in range(self.channels_number)]).T

        r['time'] = max(new_data[:,self.channels_number])
        r = r.reset_index().set_index(['time', 'index'])

        return r