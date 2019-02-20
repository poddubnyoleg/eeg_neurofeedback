# -*- coding: utf-8 -*-
import time
import numpy as np
import multiprocessing
from functools import partial


class Helmet(object):

    def streaming(self):
        pass

    def __init__(self):

        self.channels_number = 8
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
        import openbci
        self.board = openbci.OpenBCICyton(port='/dev/tty.usbserial-DM00Q4BH')
        Helmet.__init__(self)


class MuseHelmet(Helmet):

    def __init__(self):
        from muselsl import stream, list_muses
        from pylsl import StreamInlet, resolve_byprop

        self.channels_number = 4

        muses = list_muses()

        # stream module connects with Muse, but doesn't start streaming data
        muse_ble_connect = multiprocessing.Process(target=stream, args=(muses[0]['address'],))
        muse_ble_connect.start()

        streams = resolve_byprop('type', 'EEG', timeout=2)
        inlet = StreamInlet(streams[0], max_chunklen=12)

        Helmet.__init__(self)

    # Muse lsl streams data in chunks with predefined timestamps
    # so putting data in queue is in streaming func, passing insert_sample
    def streaming(self):
        # channels: TP9, AF7, AF8, TP10, Right Aux (not used)
        while True:

            eeg_data, timestamp = self.inlet.pull_chunk(
                timeout=0)

            # todo put chunk, concat eeg and time arrays
            for i in range(len(timestamp)):
                self.q.put(eeg_data[i][0:4] + [timestamp[i]] + [int(timestamp[i])])

            time.sleep(0.01)

    # todo make unique signature with inlet.pull_sample
    def insert_sample_record(self, sample):
        pass
        #self.q.put(sample.channel_data + [time.time()] + [int(time.time())])