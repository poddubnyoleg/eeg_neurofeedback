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


def button_func(app):
    app.state.tuning_phase = False


def app_runner(doc, app):
    last_time = time.time()

    while True:
        if int(time.time()) - int(last_time) >= 1:

            doc.add_next_tick_callback(app.run)
            last_time = time.time()

        time.sleep(0.01)


my_helmet = helmet.Helmet(helmet_type='fake')
my_helmet.start_stream()

visuals = bokeh_visuals.BokehVisuals()
physical_feedback = feedback_services.PhysicalFeedback()

app = feedback_services.Application(helmet=my_helmet,
                                    visuals=visuals,
                                    physical_feedback=physical_feedback,
                                    protocol_params=dict(warm_period=5,
                                                         calibration_iters=2,
                                                         calibration_halfiter_period=5,
                                                         feedback_period=10,
                                                         relax_period=5,
                                                         recalibration_period=5,
                                                         recalibration_accuracy=0.7)
                                    )


visuals.button_link(partial(button_func, app))

visuals.add_layout()

thread = Thread(target=app_runner, args=(visuals.doc, app))
thread.daemon = True  # to end session properly when terminating main thread with ctrl+c
thread.start()
