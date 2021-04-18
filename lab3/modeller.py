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
        min_rec_qs = float('Inf')

        for rec in self._receivers:
            if rec.queue.current_queue_size < min_rec_qs:
                min_rec_qs = rec.queue.current_queue_size

        for rec in self._receivers:
            if rec.queue.current_queue_size == min_rec_qs:
                rec.receive_request(time)
                return rec


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
    def __init__(self, m1, s1, m2, s2, n1, n2, ret_prob):
        # self._generator = RequestGenerator(NormalGenerator(m1, s1))
        # self._processor = RequestProcessor(NormalGenerator(m2, s2), ret_prob)
        # self._generator.add_receiver(self._processor)
        self._generators = [RequestGenerator(NormalGenerator(m1[i], s1[i])) for i in range(n1)]
        self._processors = [RequestProcessor(NormalGenerator(m2, s2), ret_prob) for i in range(n2)]

        for p in self._processors:
            for g in self._generators:
                g.add_receiver(p)

    # def set_generator(self, m, s):
    #     self._generator = RequestGenerator(NormalGenerator(m, s))
    #     self._generator.add_receiver(self._processor)
    #
    # def set_processor(self, m, s, ret_prob):
    #     self._generator.remove_receiver(self._processor)
    #     self._processor = RequestProcessor(NormalGenerator(m, s), ret_prob)
    #     self._generator.add_receiver(self._processor)

    # def event_based_modelling(self, modelling_time):
    #     generator = self._generator
    #     processor = self._processor
    #
    #     gen_period = generator.next_time()
    #     proc_period = gen_period + processor.next_time()
    #     current_time = 0
    #     while current_time < modelling_time:
    #         #         while proc_period < request_count:
    #         if gen_period <= proc_period:
    #             generator.emit_request(current_time)
    #             gen_period += generator.next_time()
    #         if gen_period >= proc_period:
    #             #                 cur_queue = processor.queue.current_queue_size
    #             processor.process(current_time)
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
        generator = self._generators[0]
        processor = self._processors[0]

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

    def time_based_modellingg(self, modelling_time, dt):
        generators = self._generators
        processors = self._processors

        gen_pers = [generators[i].next_time() for i in range(len(generators))]
        proc_pers = [-1 for i in range(len(processors))]

        proc_pers[0] = min(gen_pers) + processors[0].next_time()
        current_time = 0

        while current_time < modelling_time:
            #         while current_time < request_count:
            for i in range(len(processors)):
                if current_time >= proc_pers[i] >= 0:
                    processors[i].process(current_time)

                    if processors[i].queue.current_queue_size > 0:
                        proc_pers[i] += processors[i].next_time()
                    else:
                        proc_pers[i] = -1

            for i in range(len(generators)):
                if gen_pers[i] <= current_time:
                    proc = generators[i].emit_request(current_time)

                    proc_i = processors.index(proc)

                    if proc_pers[proc_i] == -1:
                        proc_pers[proc_i] = gen_pers[i] + processors[proc_i].next_time()

                    gen_pers[i] += generators[i].next_time()

            current_time += dt

            for i in range(len(processors)):
                processors[i].queue.recalc_avg_queue_size()

        return processors[0].queue.avg_queue_size, \
               processors[0].queue.avg_waiting_time, \
               processors[0].processed_requests
