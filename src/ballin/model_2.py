from typing import List, Optional, cast

from simpy import Process

from ballin.devices import Sensor
from ballin.env import BallinSimEnv, EnvData
from common.data import DataReceiver, DataSender, DataProcessor
from common.entity import Entity
from common.sim_env import SimEnv


class Cluster(Entity):
    sensors: "List[Sensor]"
    ground_base: "GroundBase"

    def __init__(self, env: "SimEnv"):
        super().__init__(env)
        self.sensors = []
        self.ground_base = GroundBase(env, self)


class GroundBase(DataReceiver, DataSender, DataProcessor, Entity):
    env: "BallinSimEnv"
    cluster: "Cluster"
    action: "Optional[Process]"

    def __init__(self, env: "SimEnv", cluster: "Cluster"):
        super().__init__(env)
        self.cluster = cluster
        self.action = None

    def start(self):
        self.action = self.env.process(self.run())

    def run(self):
        while True:
            data = yield self.data_inbox.get()
            data = cast(EnvData, data)
            self.log(f'收到 {data} 并加入处理队列')
            self.env.process(self.process_and_forward(data))

    def process_and_forward(self, data: "EnvData"):
        yield self.env.process(self.compute(data))
        yield self.env.process(self.send_data(data, self.env.observer))
