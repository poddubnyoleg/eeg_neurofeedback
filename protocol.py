# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import time
import numpy as np
from collections import deque
import subprocess
import multiprocessing


# те же комментарии, что и в feedback_services
class Protocol:

    @staticmethod
    def alert(alert_type='switch'):
        if alert_type == 'switch':
            audio_file = "eeg_neurofeedback/sounds/switch.wav"
        else:
            audio_file = "eeg_neurofeedback/sounds/big.wav"
        subprocess.call(["afplay", audio_file])

    # todo create smooth changer
    # почему настройки звука в протоколе?
    # почему не сделать отдельный объект, описывающий обратную связь
    # можно будет сделать и текстовую обратную связь, а потом уже сделать
    # вариант со звуком
    def change_volume(self, new_volume):
        self.volume_averaging_array.append(new_volume)
        self.volume_averaging_array.popleft()
        self.sound_volume = np.mean(self.volume_averaging_array)

    def mute_sound(self):
        self.volume_averaging_array = deque([0] * 3)
        self.sound_volume = np.mean(self.volume_averaging_array)

    '''
    Protocol settings:

    warm_period - first seconds of new task to throw away from learning
    calibration_protocol - here we set alternating task with specified periods for initial calibration
    feedback_period - passing feedback after calibration
    relax_period - relax after successful feedback session
    recalibration_period - at this point of feedback session check prediction accuracy
    recalibration_accuracy - at least prediction accuracy for feedback session to continue
    '''

    def __init__(self,
                 warm_period=5,
                 calibration_protocol=((2*60, 'relax'),
                                       (2*60, 'target'),
                                       (2*60, 'relax'),
                                       (2*60, 'target')),
                 feedback_period=10 * 60,
                 relax_period=2 * 60,
                 recalibration_period=5 * 60,
                 recalibration_accuracy=0.7
                 ):

        self.warm_period = warm_period
        self.calibration_protocol = calibration_protocol

        # remap calibration_protocol for time started
        self.run_calibration_protocol = []
        st = 0
        for i in self.calibration_protocol:
            self.run_calibration_protocol.append((st, i[1]))
            st += i[0]

        self.feedback_period = feedback_period
        self.relax_period = relax_period

        self.recalibration_period = recalibration_period
        self.recalibration_accuracy = recalibration_accuracy

        # init models for machine learning
        self.estimators = [('reduce_dim', PCA(n_components=35)),
                           ('scaling', StandardScaler()),
                           ('clf', SVC(probability=True, kernel='sigmoid', C=0.1, gamma=0.1))]
        self.clf = Pipeline(self.estimators)

        self.current_human_state = ''
        self.current_human_state_start = time.time()

        self.current_feedback_state = ''
        self.current_feedback_state_start = time.time()

        self.states = []

        # feedback sound processing
        self.volume_window = 1
        self.volume_averaging_array = deque([0]*self.volume_window)
        self.sound_volume = 0
        self.current_prediction = 0

        self.current_accuracy = 0

        # alert to start protocol
        al = multiprocessing.Process(target=self.alert, args=('big',))
        al.start()

    def fit(self, featurespace, just_score=False):
        # values - data, index - seconds
        # todo exclude warm period
        extended_states = self.states + [(time.time()*10, 'end')]
        featurespace = featurespace.unstack()

        # filter featurespace with start session time
        featurespace = featurespace[featurespace.index.values >= extended_states[0][0]]
        cs = 0
        y = []

        # todo simplify iteration
        for s in featurespace.index.values:
            if s >= extended_states[cs+1][0]:
                cs += 1

            y.append(extended_states[cs][1])

        if just_score:
            return self.clf.score(featurespace.values, y)
        else:
            self.clf.fit(featurespace.values, y)
            return self.clf.score(featurespace.values, y)

    def evaluate(self, featurespace, current_features):

        """
        Protocol states and actions:

        calibration ongoing - continue
        calibration ends - fit, start feedback with sound
        feedback on recalibration_period - check accuracy, if low - start calibration
        feedback on feedback period - start relax, stop feedback
        relax on relax period - fit, start feedback
        """
        td = time.time() - self.current_feedback_state_start
        new_human_state = self.current_human_state
        new_feedback_state = self.current_feedback_state

        if self.current_human_state == '':
            new_human_state = self.calibration_protocol[0][1]
            new_feedback_state = 'calibration'

        if self.current_feedback_state == 'calibration':
            for i in range(len(self.run_calibration_protocol)-1):
                if (td >= self.run_calibration_protocol[i][0]) & (td < self.run_calibration_protocol[i+1][0]):
                    new_human_state = self.run_calibration_protocol[i][1]
                    break
            if (td >= self.run_calibration_protocol[-1][0])&(
                    td < self.run_calibration_protocol[-1][0]+self.calibration_protocol[-1][0]):
                new_human_state = self.run_calibration_protocol[-1][1]
            elif td >= self.run_calibration_protocol[-1][0]+self.calibration_protocol[-1][0]:
                new_feedback_state = 'feedback'
                new_human_state = 'target'
                self.current_accuracy = self.fit(featurespace)

        if self.current_feedback_state == 'feedback':

            # todo check accuracy on last feedback period
            if (td >= self.recalibration_period) & (td < self.feedback_period):
                score = self.fit(featurespace, just_score=True)
                if score < self.recalibration_accuracy:
                    new_human_state = self.calibration_protocol[0][1]
                    new_feedback_state = 'calibration'
                    self.mute_sound()

            elif td >= self.feedback_period:
                new_human_state = 'relax'
                new_feedback_state = 'relax'
                self.mute_sound()
            else:

                self.current_prediction = self.clf.predict_proba(current_features.unstack().values)[0][
                    list(self.clf.classes_).index('target')]

                self.change_volume(1-self.current_prediction)

        if self.current_feedback_state == 'relax':

            if td > self.relax_period:

                self.fit(featurespace)

                new_feedback_state = 'feedback'
                new_human_state = 'target'

        if (new_human_state != self.current_human_state) | (new_feedback_state != self.current_feedback_state):
            self.states.append((time.time(), new_human_state, new_feedback_state))

            al = multiprocessing.Process(target=self.alert, args=('switch',))
            al.start()

            if new_human_state != self.current_human_state:
                self.current_human_state_start = time.time()
                self.current_human_state = new_human_state

            if new_feedback_state != self.current_feedback_state:
                self.current_feedback_state_start = time.time()
                self.current_feedback_state = new_feedback_state



