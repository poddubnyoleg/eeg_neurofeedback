from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import time
import sounddevice as sd
import numpy as np
from collections import deque
import subprocess
import multiprocessing


class Protocol:
    # todo put parameters as init options

    def alert(self, alert_type='switch'):
        if alert_type=='switch':
            audio_file = "sounds/switch.wav"
        else:
            audio_file = "sounds/big.wav"
        subprocess.call(["afplay", audio_file])

    def __init__(self):
        self.warm_period = 5   # first seconds of new task to throw away from learning

        # here we set alternating task with specified periods for initial calibration
        self.calibration_protocol = [(2*60, 'relax'),
                                     (2*60, 'target'),
                                     (2*60, 'relax'),
                                     (2*60, 'target')
                                     ]
        # remap calibration_protocol for time started
        self.run_calibration_protocol = []
        st = 0
        for i in self.calibration_protocol:
            self.run_calibration_protocol.append((st, i[1]))
            st += i[0]

        self.feedback_period = 10*60    # passing feedback after calibration
        self.relax_period = 2*60        # relax after successful feedback session

        self.recalibration_period = 5*60  # at this point of feedback session check prediction accuracy
        self.recalibration_accuracy = 0.7  # at least prediction accuracy for feedback session to continue

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
        self.volume_averaging_array = deque([0]*3)
        self.sound_volume = 0
        sd.Stream(channels=2, callback=self.sound_callback)

        # alert to start protocol

        al = multiprocessing.Process(target=self.alert, args=('big',))
        al.start()


    def change_volume(self, new_volume):
        self.volume_averaging_array.append(new_volume)
        self.volume_averaging_array.popleft()
        self.sound_volume = np.mean(self.volume_averaging_array)

    def mute_sound(self):
        self.volume_averaging_array = deque([0] * 3)
        self.sound_volume = np.mean(self.volume_averaging_array)

    def sound_callback(self, indata, outdata, frames, time, status):

        if status:
            print(status)
        outdata[:] = np.random.rand(512, 2) * self.sound_volume

    def fit(self, featurespace, just_score=False):
        # values - data, index - seconds
        # todo exclude warm period
        extendend_states = self.states + [(time.time()*10, 'end')]

        cs = 0
        y = []

        for s in featurespace.index.values:
            if s >= extendend_states[cs+1][0]:
                cs += 1

            y.append(extendend_states[cs][1])

        if just_score:
            return self.clf.score(featurespace, y)
        else:
            self.clf.fit(featurespace, y)

    def evaluate(self, featurespace, current_features):

        # calibration ongoing - continue
        # calibration ends - fit, start feedback with sound
        # feedback on recalibration_period - check accuracy, if low - start calibration
        # feedback on feedback period - start relax, stop feedback
        # relax on relax period - fit, start feedback

        # todo add sounds

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

        if self.current_feedback_state == 'feedback':

            score = self.fit(featurespace, just_score=True)

            if (td >= self.recalibration_period) & (td < self.feedback_period) & (score < self.recalibration_accuracy):
                new_human_state = self.calibration_protocol[0][1]
                new_feedback_state = 'calibration'

            elif td >= self.feedback_period:
                new_human_state = 'relax'
                new_feedback_state = 'relax'
                self.mute_sound()
            else:
                self.change_volume(1 - self.clf.predict_proba(current_features.values)[
                    list(self.clf.classes_).index('target')][0])

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



