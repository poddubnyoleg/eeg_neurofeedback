# import open_bci_connector
import multiprocessing
import feature_generation
from protocol import Protocol
from bokeh.models.sources import ColumnDataSource
from bokeh.models import LinearColorMapper
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import TextInput, Button
from bokeh.models.glyphs import Text
from bokeh.models import LinearAxis, Range1d
from functools import partial
from threading import Thread
import pandas as pd
import sounddevice as sd
import csv

import numpy as np
import time

# todo put in script parameter
fake_data = True

if fake_data:

    def handle_sample():
        global q
        q.put([np.random.rand() for r in range(8)] + [time.time()] + [int(time.time())])


    def streaming():
        while True:
            handle_sample()
            time.sleep(0.01)
else:

    def handle_sample(sample):
        global q
        q.put(sample.channel_data + [time.time()] + [sample.id])
        # todo log_id for headset data

    def streaming():
        open_bci_connector.board.start_streaming(handle_sample)


# todo add port selector for real data
# create charts for testing headset
figs = [figure(plot_width=300, plot_height=100, toolbar_location=None) for i in range(8)]
for f in figs:
    f.extra_y_ranges = {"foo": Range1d(start=-1, end=1)}

[f.add_layout(LinearAxis(y_range_name="foo"), 'right') for f in figs]

cds = ColumnDataSource(data={i: [] for i in [item for sublist in
                                             [['y' + str(i), 'f' + str(i)] for i in range(8)] for item in sublist] +
                             ['x']})

lines = [(figs[i].line('x', 'y' + str(i), source=cds, line_alpha=0.3),
          figs[i].line('x', 'f' + str(i), source=cds, line_color='black', y_range_name="foo", line_width=1.3)) for i in
         range(8)]

for f in figs:
    f.axis.visible = False

# create charts for visualising features
cds_feats_data = {'x': [],
                  'y': []}
for i in range(8):
    cds_feats_data['value_' + str(i)] = []

cds_feats = ColumnDataSource(data=cds_feats_data)

colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=-5, high=5)

figs_feats = [figure(plot_width=300, plot_height=100, toolbar_location=None) for i in range(8)]
for f in figs_feats:
    f.axis.visible = False
[figs_feats[i].rect(x="x", y="y", width=1, height=1, source=cds_feats,
                    line_color=None, fill_color={'field': 'value_' + str(i), 'transform': mapper}) for i in range(8)]

# create figure for forecast and status
forecast_status_cds = ColumnDataSource(data={'x': [],
                                             'stage': [],
                                             'prediction': []})
forecast_status_figure = figure(plot_width=300, plot_height=100, toolbar_location=None)
forecast_status_figure.line('x', 'stage', source=forecast_status_cds, line_color='red', line_alpha=0.5)
forecast_status_figure.line('x', 'prediction', source=forecast_status_cds, line_color='black')
forecast_status_figure.axis.visible = False

# figure for status
status_text_cds = ColumnDataSource(data={'x': [0], 'y': [0], 'text': ['Calibration']})
status_text_figure = figure(plot_width=300, plot_height=100, toolbar_location=None)
status_text_figure.axis.visible = False
status_text = Text(x="x", y="y", text="text", text_color='black')
status_text_figure.add_glyph(status_text_cds, status_text)

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
                        calibration_protocol=((5, 'relax'),
                                              (5, 'target'),
                                              (5, 'relax'),
                                              (5, 'target')),
                        feedback_period=10,
                        relax_period=5,
                        recalibration_period=5,
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

    cds.stream(update_data, 300)

    if not test_phase:
        update_data = {'x': nfetd.index.get_level_values(0).values,
                       'y': nfetd.index.get_level_values(1).values}

        for i in range(8):
            update_data['value_' + str(i)] = nfetd[i].values

        cds_feats.stream(update_data, len(nfetd.columns)*8*100)

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
q = multiprocessing.Queue()
p = multiprocessing.Process(target=streaming)
p.start()

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
            features_data = features_data.append(new_features_data)

            # update charts
            doc.add_next_tick_callback(partial(update_test_charts, nd=nd, sd=ses_data, nfd=new_filtered_data,
                                               nfetd=new_features_data))

            new_data = []
            last_time = t

        time.sleep(0.001)


# create layout
# todo add stop button
# todo add logging
update = Button(label="Start session")
update.on_click(starter)
inputs = widgetbox([update], width=300)

doc = curdoc()

layout = column(row(figs[0:2] + figs_feats[0:2]), row(figs[2:4] + figs_feats[2:4]),
                row(figs[4:6] + figs_feats[4:6]), row(figs[6:]+figs_feats[6:]),
                row(inputs, status_text_figure, forecast_status_figure))
doc.add_root(layout)

thread = Thread(target=updater)
thread.daemon = True  # to end session properly when terminating main thread with ctrl+c
thread.start()


