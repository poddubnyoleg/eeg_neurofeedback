# -*- coding: utf-8 -*-
import feature_generation
import numpy as np
import pandas as pd
import sounddevice as sd
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import time


class PhysicalFeedback:
    # @staticmethod
    def sound_callback(self, indata, outdata, frames, time, status):
        outdata[:] = np.random.rand(512, 2) * self.sound_volume

    def __init__(self):
        self.sound_volume = 0
        stream = sd.Stream(channels=2, callback=self.sound_callback)
        stream.start()


def get_data(helmet):
    new_data = []
    while not helmet.q.empty():
        new_data.append(helmet.q.get())
    return new_data


class TuningState:

    def __init__(self, helmet, visuals, physical_feedback,
                 # warm_period=5,
                 # calibration_iters=2,
                 # calibration_halfiter_period=2 * 60,
                 # feedback_period=10 * 60,
                 # relax_period=2 * 60,
                 # recalibration_period=5 * 60,
                 # recalibration_accuracy=0.7
                 protocol_params):
        self.helmet = helmet
        self.visuals = visuals
        self.physical_feedback = physical_feedback
        self.protocol_params = protocol_params
        self.online_filters = [
            feature_generation.OnlineFilter(fs=250, notch_f0=50, notch_q=30, low_cut=1, high_cut=40, order=5) for
            i in range(8)]
        self.tuning_phase = True
        self.filtered_data = np.ndarray(shape=(0, 9))

    def run(self):
        new_data = np.array(get_data(self.helmet))
        # [:,None] - to make arrays the same shape for hstack
        new_filtered_data = np.hstack([np.array([self.online_filters[i].filter(new_data[:, i])
                                                 for i in range(8)]).T, new_data[:, 9][:, None]])
        self.filtered_data = np.append(self.filtered_data, new_filtered_data, axis=0)

        self.visuals.update_tuning(new_data, new_filtered_data, self.filtered_data)

        if ~self.tuning_phase:
            return CalibrationRelax(helmet=self.helmet,
                                    visuals=self.visuals,
                                    online_filters=self.online_filters,
                                    physical_feedback=self.physical_feedback,
                                    protocol_params=self.protocol_params,
                                    filtered_data=self.filtered_data,
                                    features_data=pd.DataFrame(),
                                    ml=MachineLearning(),
                                    states_history=[],
                                    calibration_iter=1,
                                    last_time_run=time.time()
                                    )
        return self


class MachineLearning:

    def __init__(self):
        estimators = [('reduce_dim', PCA(n_components=35)),
                      ('scaling', StandardScaler()),
                      ('clf', SVC(probability=True, kernel='sigmoid', C=0.1, gamma=0.1))]
        self.clf = Pipeline(estimators)

    def fit(self, featurespace, just_score=False):
        # values - data, index - seconds
        # todo exclude warm period
        extended_states = self.states + [(time.time() * 10, 'end')]
        featurespace = featurespace.unstack()

        # filter featurespace with start session time
        featurespace = featurespace[featurespace.index.values >= extended_states[0][0]]
        cs = 0
        y = []

        # todo simplify iteration
        for s in featurespace.index.values:
            if s >= extended_states[cs + 1][0]:
                cs += 1

            y.append(extended_states[cs][1])

        if just_score:
            return self.clf.score(featurespace.values, y)
        else:
            self.clf.fit(featurespace.values, y)
            return self.clf.score(featurespace.values, y)


class ProtocolCommonState:

    def __init__(self, helmet, visuals, online_filters, physical_feedback, protocol_params,
                 filtered_data, features_data, ml, states_history, calibration_iter,
                 last_time_run):
        self.helmet = helmet,
        self.visuals = visuals,
        self.online_filters = online_filters
        self.physical_feedback = physical_feedback
        self.protocol_params = protocol_params
        self.filtered_data = filtered_data
        self.features_data = features_data
        self.ml = ml
        self.states_history = states_history
        self.state_start = time.time()
        self.calibration_iter = calibration_iter
        self.last_time_run = last_time_run

        self.params_to_pass = dict(helmet=self.helmet,
                                   visuals=self.visuals,
                                   online_filters=self.online_filters,
                                   physical_feedback=self.physical_feedback,
                                   protocol_params=self.protocol_params,
                                   filtered_data=self.filtered_data,
                                   features_data=self.features_data,
                                   ml=self.ml,
                                   states_history=self.states_history,
                                   calibration_iter=self.calibration_iter,
                                   last_time_run=self.last_time_run
                                   )

    def update_data_charts(self):
        new_data = np.array(get_data(self.helmet))
        # [:,None] - to make arrays the same shape for hstack
        new_filtered_data = np.hstack([np.array([self.online_filters[i].filter(new_data[:, i])
                                                 for i in range(8)]).T, new_data[:, 9][:, None]])
        self.filtered_data = np.append(self.filtered_data, new_filtered_data, axis=0)
        new_features_data = pd.DataFrame(
            self.filtered_data[(np.where(self.filtered_data[:, 8] > int(self.last_time_run) - 1)) &
                               (np.where(self.filtered_data[:, 8] <= int(time.time()) - 1))]
            ).groupby(8).apply(
            lambda x: pd.DataFrame([feature_generation.spectral_features(x[i]) for i in range(8)]).T)

        self.features_data = self.features_data.append(new_features_data)

        self.visuals.update_tuning(new_data, new_filtered_data, self.filtered_data)
        self.visuals.update_protocol(new_data)


class CalibrationRelax(ProtocolCommonState):

    def run(self):
        self.update_data_charts()
        self.last_time_run = time.time()
        if time.time()-self.state_start > self.protocol_params['calibration_halfiter_period']:
            return CalibrationTarget(**self.params_to_pass)
        return self


class CalibrationTarget(ProtocolCommonState):

    def run(self):
        self.update_data_charts()
        self.last_time_run = time.time()

        if time.time()-self.state_start > self.protocol_params['calibration_halfiter_period']:
            if self.calibration_iter < self.protocol_params['calibration_iters']:
                self.calibration_iter += 1
                return CalibrationRelax(**self.params_to_pass)
            else:
                return FeedbackTarget(**self.params_to_pass)
        return self


class FeedbackRelax(ProtocolCommonState):

    def run(self):
        pass


class FeedbackTarget(ProtocolCommonState):

    def run(self):
        pass


class Application:

    def __init__(self, helmet, visuals, physical_feedback, protocol_params):
        self.state = TuningState(helmet, visuals, physical_feedback, protocol_params)

    def run(self):
        self.state = self.state.run()
