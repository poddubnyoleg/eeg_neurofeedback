# -*- coding: utf-8 -*-
import time
import numpy as np
import multiprocessing


class Helmet:

    def __init__(self, helmet_type='fake'):

        self.q = multiprocessing.Queue()

        if helmet_type == 'fake':

            def handle_sample():
                self.q.put([np.random.rand() for r in range(8)] + [time.time()] + [int(time.time())])

            def streaming():
                while True:
                    handle_sample()
                    time.sleep(0.01)

        elif helmet_type == 'cython':
            import open_bci_connector

            def handle_sample(sample):
                self.q.put(sample.channel_data + [time.time()] + [int(time.time())])
                # todo log_id for headset data

            def streaming():
                open_bci_connector.board.start_streaming(handle_sample)

        self.p = multiprocessing.Process(target=streaming)

    def start_stream(self):
        self.p.start()

    def get_data(self):
        new_data = []
        while not self.q.empty():
            new_data.append(self.q.get())
        return new_data
