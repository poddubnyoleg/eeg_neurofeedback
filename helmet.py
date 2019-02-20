# -*- coding: utf-8 -*-
import time
import numpy as np
import multiprocessing
from functools import partial
import openbci


class Helmet(object):

    def streaming(self):
        pass

    def __init__(self):

        self.q = multiprocessing.Queue()
        self.p = multiprocessing.Process(target=self.streaming)

    def insert_sample_record(self, sample):
        self.q.put(sample.channel_data + [time.time()] + [int(time.time())])

    def start_stream(self):
        self.p.start()

    def get_data(self):
        new_data = []
        while not self.q.empty():
            new_data.append(self.q.get())
        return new_data


class FakeHelmetSample(object):

    def __init__(self):
        self.channel_data = [np.random.rand() for r in range(8)]


class FakeHelmet(Helmet):

    def streaming(self):
        while True:
            self.insert_sample_record(FakeHelmetSample())
            time.sleep(0.01)


class CytonHelmet(Helmet):

    def streaming(self):
        self.board.start_streaming(callback=self.insert_sample_record)

    def __init__(self):
        self.board = openbci.OpenBCICyton(port='/dev/tty.usbserial-DM00Q4BH')
        Helmet.__init__(self)


