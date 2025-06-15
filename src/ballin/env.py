from typing import Optional, cast, Dict

from simpy import Process

from ballin.devices import Sensor
from common.data import TransmittableData, ComputableData, DataReceiver
from common.entity import Entity
from common.sim_env import SimEnv


class BallinSimEnv(SimEnv):
    observer: "Observer"

    def __init__(self):
        super().__init__()
        self.observer = Observer(self)


class EnvData(TransmittableData, ComputableData, Entity):
    sampled_at: float

    @property
    def aoi(self):
        return self.env.now - self.sampled_at

    def __init__(self, env: "SimEnv", sensor: "Sensor", size: float):
        super().__init__(env, transient=True)
        self.sampled_by = sensor
        self.sampled_at = float(env.now)
        self.trans_size = size
        self.process_size = size


class Observer(DataReceiver, Entity):
    aoi: "Dict[Sensor, float]"
    watch_action: "Optional[Process]"

    def __init__(self, env: 'SimEnv'):
        super().__init__(env)
        self.aoi = dict()
        self.watch_action = None

    @property
    def aoi_sum(self) -> float:
        return sum(self.aoi.values())

    def start(self):
        self.watch_action = self.env.process(self.watch())

    def watch(self):
        while True:
            data = yield self.data_inbox.get()
            data = cast(EnvData, data)
            self.log(f'|> 收到 {data}')

    def plot(self):
        raise NotImplementedError
