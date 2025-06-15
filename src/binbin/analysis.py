from matplotlib import gridspec
from matplotlib import pyplot as plt

from binbin.gym_env import Observation


class Analyzer:
    def __init__(self):
        self.timestamp = []
        self.cpu_queue_per_core = []
        self.cpu_free = []

    def add_observation(self, obs: "Observation"):
        self.timestamp.append(obs['now'])
        self.cpu_queue_per_core.append(obs['cpu_queue_span_per_core'])
        self.cpu_free.append(obs['cpu_free'])

    def show(self):
        fig = plt.figure(tight_layout=True)
        fig.suptitle("Resource Usage")

        gs = gridspec.GridSpec(2, 1)

        ax = fig.add_subplot(gs[0, 0])
        ax.step(self.timestamp, self.cpu_queue_per_core)
        ax.set_xlabel('time')
        ax.set_ylabel('CPU queue per core')

        ax = fig.add_subplot(gs[1, 0])
        ax.step(self.timestamp, self.cpu_free)
        ax.set_xlabel('time')
        ax.set_ylabel('free CPUs')

        plt.show()
