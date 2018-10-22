#import open_bci_connector
import multiprocessing
import feature_generation
from bokeh.models.sources import ColumnDataSource
from bokeh.models import LinearColorMapper
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import TextInput, Button
from bokeh.models import LinearAxis, Range1d
from bokeh.transform import transform
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

cds = ColumnDataSource(data={i: [] for i in [item for sublist in
                                            [['y'+str(i), 'f'+str(i)] for i in range(8)] for item in sublist]+['x']})

lines = [(figs[i].line('x', 'y'+str(i), source=cds, line_alpha=0.5),
          figs[i].line('x', 'f'+str(i), source=cds, line_color='red', y_range_name="foo", line_width=2)) for i in range(8)]
for f in figs:
    f.axis.visible = False

online_filters = [feature_generation.OnlineFilter(fs=250, notch_f0=50, notch_q=30, low_cut=1, high_cut=40, order=5) for
                  i in range(8)]

test_phase = True


def starter():
    global test_phase
    test_phase = False

    # redraw figs with heatmaps
    cds2 = ColumnDataSource({'value': np.random.rand(1000), 'y': np.tile(np.arange(0, 10, 1), 100),
                            'x': np.repeat(np.arange(0, 100, 1), 10)})
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=10, high=50)

    fig2 = figure(plot_width=500, plot_height=100, toolbar_location=None)
    fig2.rect(x="x", y="y", width=1, height=1, source=cds2,
           line_color=None, fill_color={'field': 'value', 'transform': mapper})

    fig2.axis.axis_line_color = None
    fig2.axis.major_tick_line_color = None
    fig2.axis.major_label_text_font_size = "5pt"
    fig2.axis.major_label_standoff = 0
    fig2.xaxis.major_label_orientation = 1.0

    layout.children[0].children[0] = fig2


update = Button(label="Start session")
update.on_click(starter)
inputs = widgetbox([update], width=200)

doc = curdoc()

layout = column(row(figs[0:2]), row(figs[2:4]), row(figs[4:6]), row(figs[6:]), inputs)
doc.add_root(layout)


def update_test_charts(nd, sd):
    nd = np.array(nd)
    lsd = len(sd)   # calculate before, as sd can change during stream

    update_data = {'x': np.arange(lsd - len(nd[:, 0]), lsd, 1)}

    for i in range(8):
        update_data['f'+str(i)] = online_filters[i].filter(nd[:, i])
        update_data['y'+str(i)] = nd[:, i]

    cds.stream(update_data, 300)


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

            new_data = []


        time.sleep(0.001)


thread = Thread(target=updater)
thread.daemon = True    # to end session properly when terminating main thread with ctrl+c
thread.start()


# todo on click replace with spectras
# todo on click calculate initial dataset and start update it <- and start store features dataset
# todo log_id for headset data






