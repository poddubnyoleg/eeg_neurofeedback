# -*- coding: utf-8 -*-


import feature_generation
from protocol import Protocol

from functools import partial
from threading import Thread
import pandas as pd
import sounddevice as sd
import csv

import numpy as np
import time
import bokeh_visuals
import helmet


my_helmet = helmet.Helmet(helmet_type='fake')
visuals = bokeh_visuals.BokehVisuals()


# init filters
online_filters = [feature_generation.OnlineFilter(fs=250, notch_f0=50, notch_q=30, low_cut=1, high_cut=40, order=5) for
                  i in range(8)]

test_phase = True
filtered_data = np.ndarray(shape=(0, 9))
features_data = pd.DataFrame()

# start feedback stream
sound_volume = 0

def sound_callback(indata, outdata, frames, time, status):
    outdata[:] = np.random.rand(512, 2) * sound_volume


stream = sd.Stream(channels=2, callback=sound_callback)
stream.start()


def starter():
    global test_phase
    global online_filters
    global features_data
    global protocol
    global sound_volume

    test_phase = False

    # set initial features data and cds
    # features data - df, ind - (sec, freq number), cols - electrodes
    features_data = pd.DataFrame(filtered_data).groupby(8).apply(
        lambda x: pd.DataFrame([feature_generation.spectral_features(x[i]) for i in range(8)]).T)

    cds_data = {'x': features_data.index.get_level_values(0).values,
                'y': features_data.index.get_level_values(1).values}
    for i in range(8):
        cds_data['value_' + str(i)] = features_data[i].values

    protocol = Protocol(warm_period=5,
                        calibration_protocol=((30, 'relax'),
                                              (30, 'target'),
                                              (30, 'relax'),
                                              (30, 'target')),
                        feedback_period=100,
                        relax_period=50,
                        recalibration_period=50,
                        recalibration_accuracy=0.7)

    sound_volume = protocol.sound_volume


def update_test_charts(nd, sd, nfd, nfetd):
    global protocol
    global sound_volume

    lsd = len(sd)

    update_data = {'x': np.arange(lsd - len(nd[:, 0]), lsd, 1)}

    for i in range(8):
        update_data['f' + str(i)] = nfd[:, i]
        update_data['y' + str(i)] = nd[:, i]

    cds.stream(update_data, 1000)

    if not test_phase:
        update_data = {'x': nfetd.index.get_level_values(0).values,
                       'y': nfetd.index.get_level_values(1).values}

        for i in range(8):
            update_data['value_' + str(i)] = nfetd[i].values

        cds_feats.stream(update_data, len(nfetd.columns)*8*1000)

        protocol.evaluate(features_data, nfetd)
        sound_volume = protocol.sound_volume

        # update feedback status chart
        if protocol.current_human_state == 'target':
            stage = 1
        else:
            stage = 0

        if (protocol.current_human_state == 'target') & (protocol.current_feedback_state == 'feedback'):
            prediction = protocol.current_prediction
        else:
            prediction = 0

        forecast_status_cds.stream({'x': [int(time.time())-1], 'stage': [stage], 'prediction': [prediction]}, 1000)

        # update current accuracy
        if protocol.current_feedback_state == 'calibration':
            status_text_cds.stream({'x': [0], 'y': [0], 'text': ['calibration']}, 1)
        else:
            status_text_cds.stream({'x': [0], 'y': [0], 'text': ['feedback (accuracy ' +
                                                                 str(round(protocol.current_accuracy, 3))]}, 1)


ses_data = []

my_helmet.start_stream()

# logging init
raw_data_writer = csv.writer(open('raw_data.csv', 'wb'), delimiter=';')


def updater():
    global filtered_data
    global features_data

    last_time = time.time()
    new_data = []

    while True:

        while not q.empty():
            new_q = q.get()

            new_data.append(new_q)
            ses_data.append(new_q)

            raw_data_writer.writerow(new_q)

        if int(time.time()) - int(last_time) >= 1:

            # todo if process consumes more than 1 sec - alert
            t = time.time()

            nd = np.array(new_data)

            # [:,None] - to make arrays the same shape for hstack
            new_filtered_data = np.hstack([np.array([online_filters[i].filter(nd[:, i])
                                                     for i in range(8)]).T, nd[:, 9][:, None]])
            filtered_data = np.append(filtered_data, new_filtered_data, axis=0)

            # get features of last full second

            new_features_data = pd.DataFrame(filtered_data[np.where(filtered_data[:, 8] == int(last_time))]
                                             ).groupby(8).apply(
                lambda x: pd.DataFrame([feature_generation.spectral_features(x[i]) for i in range(8)]).T)

            # todo faster update via iloc with indexes of df_
            # todo start creating features since session start
            features_data = features_data.append(new_features_data)
            # update charts
            doc.add_next_tick_callback(partial(update_test_charts, nd=nd, sd=ses_data, nfd=new_filtered_data,
                                               nfetd=new_features_data))

            new_data = []
            last_time = t

        time.sleep(0.01)




visuals.button_link(feedback_service.start)
visuals.add_layout()

thread = Thread(target=updater)
thread.daemon = True  # to end session properly when terminating main thread with ctrl+c
thread.start()


