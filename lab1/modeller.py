import numpy.random as nr


class UniformGenerator:
    def __init__(self, a, b):
        if not 0 <= a <= b:
            raise ValueError('Параметры должны удовлетворять условию 0 <= a <= b')
        self._a = a
        self._b = b

    def next(self):
        return nr.uniform(self._a, self._b)


class PoissonGenerator:
    def __init__(self, la):
        self._la = la

    def next(self):
        return nr.poisson(self._la)


class NormalGenerator:
    def __init__(self, m, s):
        self._m = m
        self._s = s

    def next(self):
        return nr.normal(self._m, self._s)


class RequestGenerator:
    def __init__(self, generator):
        self._generator = generator
        self._receivers = set()

    def add_receiver(self, receiver):
        self._receivers.add(receiver)

    def remove_receiver(self, receiver):
        try:
            self._receivers.remove(receiver)
        except KeyError:
            pass

    def next_time(self):
        return self._generator.next()

    def emit_request(self, time):
        for rec in self._receivers:
            rec.receive_request(time)


class Queue:
    def __init__(self):
        self._current_queue_size = 0
        self._max_queue_size = 0
        self._avg_queue_size = 0
        self._avg_recalcs = 0
        self._avg_waiting_time = 0
        self._time_recalcs = 0
        self._arrive_times = []

    @property
    def max_queue_size(self):
        return self._max_queue_size

    @property
    def current_queue_size(self):
        return self._current_queue_size

    @property
    def avg_queue_size(self):
        return self._avg_queue_size

    @property
    def avg_waiting_time(self):
        return self._avg_waiting_time

    def add(self, time):
        self._current_queue_size += 1
        self._arrive_times.append(time)

    def remove(self, time):
        self._current_queue_size -= 1
        arr_time = self._arrive_times.pop(0)
        wait_time = time - arr_time
        old_cnt = self._avg_waiting_time * self._time_recalcs
        self._time_recalcs += 1
        old_cnt += wait_time
        self._avg_waiting_time = old_cnt / self._time_recalcs

    def increase_size(self):
        self._max_queue_size += 1

    def recalc_avg_queue_size(self):
        old_cnt = self._avg_queue_size * self._avg_recalcs
        self._avg_recalcs += 1
        old_cnt += self._current_queue_size
        self._avg_queue_size = old_cnt / self._avg_recalcs


class RequestProcessor(RequestGenerator):
    def __init__(self, generator, return_probability):
        super().__init__(generator)
        self._generator = generator
        self._processed_requests = 0
        self._return_probability = return_probability
        self._reentered_requests = 0
        self._queue = Queue()

    @property
    def processed_requests(self):
        return self._processed_requests

    @property
    def reentered_requests(self):
        return self._reentered_requests

    @property
    def queue(self):
        return self._queue

    def process(self, time):
        if self._queue.current_queue_size > 0:
            self._processed_requests += 1
            self._queue.remove(time)
            self.emit_request(time)
            if nr.random_sample() < self._return_probability:
                self._reentered_requests += 1
                self.receive_request(time)

    def receive_request(self, time):
        self._queue.add(time)
        if self._queue.current_queue_size > self._queue.max_queue_size:
            self._queue.increase_size()

    def next_time_period(self):
        return self._generator.next()


class Model:
    def __init__(self, a, b, la, ret_prob):
        self._generator = RequestGenerator(UniformGenerator(a, b))
        self._processor = RequestProcessor(PoissonGenerator(la), ret_prob)
        self._generator.add_receiver(self._processor)

    def set_generator(self, m, s):
        self._generator = RequestGenerator(NormalGenerator(m, s))
        self._generator.add_receiver(self._processor)

    def set_processor(self, m, s, ret_prob):
        self._generator.remove_receiver(self._processor)
        self._processor = RequestProcessor(NormalGenerator(m, s), ret_prob)
        self._generator.add_receiver(self._processor)

    # def event_based_modelling(self, request_count):
    #     generator = self._generator
    #     processor = self._processor
    #
    #     gen_period = generator.next_time()
    #     proc_period = gen_period + processor.next_time()
    #     while processor.processed_requests < request_count:
    #         #         while proc_period < request_count:
    #         if gen_period <= proc_period:
    #             generator.emit_request()
    #             gen_period += generator.next_time()
    #         if gen_period >= proc_period:
    #             #                 cur_queue = processor.queue.current_queue_size
    #             processor.process()
    #             #                 if processor.queue.current_queue_size == cur_queue:
    #             #                     request_count += 1
    #             if processor.queue.current_queue_size > 0:
    #                 proc_period += processor.next_time()
    #             else:
    #                 proc_period = gen_period + processor.next_time()
    #
    #     return (processor.processed_requests, processor.reentered_requests,
    #             processor.queue.max_queue_size, proc_period)

    def time_based_modelling(self, modelling_time, dt):
        generator = self._generator
        processor = self._processor

        gen_period = generator.next_time()
        proc_period = gen_period + processor.next_time()
        current_time = 0
        while current_time < modelling_time:
            #         while current_time < request_count:
            if current_time >= proc_period:
                # print('proc')
                #                 cur_queue = processor.queue.current_queue_size
                # print(processor.processed_requests)
                processor.process(current_time)
                # print(processor.processed_requests)
                #                 if processor.queue.current_queue_size == cur_queue:
                #                     request_count += 1
                if processor.queue.current_queue_size > 0:
                    proc_period += processor.next_time()
                else:
                    proc_period = gen_period + processor.next_time()
            if gen_period <= current_time:
                # print('gen')
                generator.emit_request(current_time)
                gen_period += generator.next_time()
            # print(processor.queue.max_queue_size, processor.queue.current_queue_size)
            current_time += dt
            processor.queue.recalc_avg_queue_size()

        return processor.queue.avg_queue_size, processor.queue.avg_waiting_time
