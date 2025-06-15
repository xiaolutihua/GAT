import random
from simpy import Process
from typing import Optional

from ballin.env import EnvData
from common.data import DataReceiver, DataSender
from common.entity import Entity
from common.sim_env import SimEnv


class Sensor(DataSender, Entity):
    cluster: "Cluster"
    sampling_interval: float
    action: "Optional[Process]"

    def __init__(self, env: "SimEnv", cluster: "Cluster"):
        super().__init__(env)
        self.cluster = cluster
        self.cluster.sensors.append(self)
        self.sampling_interval = random.uniform(1, 2)
        self.action = None

    def start(self):
        self.action = self.env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.sampling_interval)
            data = yield self.env.process(self.sample())
            yield self.env.process(self.transfer_data(data))

    def sample(self):
        self.log("=>")
        yield self.env.timeout(random.uniform(0, 1))
        data = EnvData(self.env, self, random.uniform(20, 40))
        self.log(f"<= {data}")
        return data

    def transfer_data(self, data: "EnvData"):
        self.log(f"=> {data}")
        target = self.cluster.ground_base
        yield self.env.process(self.send_data(data, target))
        self.log(f"<= {data}")

    def get_conn_delay(self, src: "DataSender", dst: "DataReceiver") -> float:
        return 0.

    def get_conn_bandwidth(self, src: "DataSender", dst: "DataReceiver") -> float:
        return float('inf')
