from scipy.signal import butter, lfilter, iirnotch, lfilter_zi
from matplotlib import mlab
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


fs_Hz = 250.0
NFFT = 256*2  # pitck the length of the fft
FFTstep = 1.*fs_Hz  # do a new FFT every half second
overlap = NFFT - FFTstep  # half-second steps
f_lim_Hz = [0, 40]   # frequency limits for plotting


def get_features(x):
    spec_PSDperHz, freqs, t = mlab.specgram(x,
                                            NFFT=NFFT,
                                            window=mlab.window_hanning,
                                            Fs=fs_Hz,
                                            noverlap=overlap
                                            )
    keep_columns = [f for f in freqs if (f >= f_lim_Hz[0] and f <= f_lim_Hz[1])]
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)
    X = pd.DataFrame(10 * np.log10(spec_PSDperBin), index=freqs).T
    X = X[keep_columns]
    X['second'] = [int(round(i)) for i in t]

    return X

