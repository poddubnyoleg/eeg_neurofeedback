# -*- coding: utf-8 -*-
from threading import Thread
import pandas as pd
import sounddevice as sd
import csv

import numpy as np
import time
import bokeh_visuals
import helmet
import feedback_services
from functools import partial
import sys


def button_func(app):
    app.state.tuning_phase = False


def app_runner(doc, app):
    last_time = time.time()

    while True:
        if int(time.time()) - int(last_time) >= 1:

            doc.add_next_tick_callback(app.run)
            last_time = time.time()

        time.sleep(0.01)


# Helmet type is gotten via sys.argv options
# Current helmets available: FakeHelmet, CytonHelmet

if len(sys.argv)>1:
    if sys.argv[1] == 'FakeHelmet':
        my_helmet = helmet.FakeHelmet()
    elif sys.argv[1] == 'CytonHelmet':
        my_helmet = helmet.CytonHelmet()
    elif sys.argv[1] == 'MuseHelmet':
        my_helmet = helmet.MuseHelmet()
    else:
        raise NameError('\n Proper helmet type is not defined. Available options: FakeHelmet, CytonHelmet, MuseHelmet \n')
else:
    print('\n No helmet is defined. FakeHelmet is used by default \n')
    my_helmet = helmet.FakeHelmet()

my_helmet.start_stream()

visuals = bokeh_visuals.BokehVisuals(my_helmet)

physical_feedback = feedback_services.PhysicalFeedback()

app = feedback_services.Application(helmet=my_helmet,
                                    visuals=visuals,
                                    physical_feedback=physical_feedback,
                                    protocol_params=dict(warm_period=5,
                                                         calibration_iters=2,
                                                         calibration_halfiter_period=60,
                                                         feedback_period=120,
                                                         relax_period=30,
                                                         recalibration_period=100,
                                                         recalibration_accuracy=0.7)
                                    )


visuals.button_link(partial(button_func, app))

visuals.add_layout()

thread = Thread(target=app_runner, args=(visuals.doc, app))
thread.daemon = True  # to end session properly when terminating main thread with ctrl+c
thread.start()

