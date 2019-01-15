# -*- coding: utf-8 -*-
import time
import numpy as np
import multiprocessing


class Helmet:

    def __init__(self, helmet_type='fake'):

        self.q = multiprocessing.Queue()

        # TODO: helmet_type намекает на то, что хорошо бы сделать это отдельными
        #       классами: FakeHelmet(Helmet) и OpenBciHelmet(Helmet)
        #       и явно передавать его приложение
        #       так будет понятен интерфейс
        if helmet_type == 'fake':

            def handle_sample():
                # формат трудно дебажить, 8 чисел в интервале [0, 1], плюс два времени в разных форматах
                # опять же, для ясности лучше сделать какой-нибудь класс Record с названиями полей
                # советую библиотеку attrs или можешь namedtuple сделать, если лень разбираться
                # но придется писать неинтересное тогда
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

# интерфейс
# helmet.start_stream()
# helmet.get_data()
# не слишком много говорит о содержании
# лучше использовать более длинные говорящие названия
# helmet.get_eeg_measurements() например
