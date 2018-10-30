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
