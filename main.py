#import open_bci_connector
import multiprocessing
import feature_generation
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import TextInput, Button
from bokeh.models import LinearAxis, Range1d
from functools import partial
from threading import Thread
from tornado import gen

import numpy as np
import time

fake_data = True

if fake_data:

    def handle_sample():
        global q
        q.put([np.random.rand() for r in range(8)] + [time.time()] + [int(time.time())])


    def streaming():
        while True:
            handle_sample()
            time.sleep(0.5)
else:

    def handle_sample(sample):
        global q
        q.put(sample.channel_data+[time.time()]+[sample.id])


    def streaming():
        open_bci_connector.board.start_streaming(handle_sample)


figs = [figure(plot_width=500, plot_height=100, toolbar_location=None) for i in range(8)]
for f in figs:
    f.extra_y_ranges = {"foo": Range1d(start=-1, end=1)}

[f.add_layout(LinearAxis(y_range_name="foo"), 'right') for f in figs]
cds = [ColumnDataSource(data=dict(x=[0], y=[0], f=[0])) for i in range(8)]
lines = [(figs[i].line('x', 'y', source=cds[i], line_alpha=0.5),
          figs[i].line('x', 'f', source=cds[i], line_color='red', y_range_name="foo", line_width=2)) for i in range(8)]
for f in figs:
    f.axis.visible = False

online_filters = [feature_generation.OnlineFilter(fs=250, notch_f0=50, notch_q=30, low_cut=1, high_cut=40, order=5) for
                  i in range(8)]

test_phase = True


def starter():
    global test_phase
    test_phase = False


update = Button(label="Start session")
update.on_click(starter)
inputs = widgetbox([update], width=200)

doc = curdoc()

doc.add_root(column(row(figs[0:2]), row(figs[2:4]), row(figs[4:6]), row(figs[6:]), inputs))


def update_test_charts(nd, sd):
    nd = np.array(nd)
    lsd = len(sd)   # calculate before, as sd can change during stream

    for i in range(8):

        cds[i].stream({'y': nd[:, i],
                       'f': online_filters[i].filter(nd[:, i]),
                       'x': np.arange(lsd - len(nd[:, i]), lsd, 1)
                       }, 300)


ses_data = []
q = multiprocessing.Queue()
p = multiprocessing.Process(target=streaming)
p.start()


def updater():

    time.sleep(3)   # wait till tornado settles to avoid data mess

    last_time = time.time()
    new_data = []

    while True:

        while not q.empty():
            new_q = q.get()

            new_data.append(new_q)
            ses_data.append(new_q)

        if int(time.time())-int(last_time) >= 1:

            last_time = time.time()
            if test_phase:
                doc.add_next_tick_callback(partial(update_test_charts, nd=new_data, sd=ses_data))

                print '---------new_line'
                print cds[0].to_df()['x'].values
                print cds[7].to_df()['x'].values

            new_data = []


        time.sleep(0.001)


thread = Thread(target=updater)
thread.daemon = True    # to end session properly when terminating main thread with ctrl+c
thread.start()


# todo custom spectrogram for group by
# todo log_id for headset data
# todo git commit





