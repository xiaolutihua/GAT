from abc import abstractmethod
from collections import defaultdict
from typing import List

from simpy import Resource, Store, Container
from simpy.resources.base import BaseResource

from common.entity import Entity
from common.sim_env import SimEnv
from constants import hstr, M, G


def stats(res: "BaseResource") -> str:
    if isinstance(res, Resource):
        return f"(用/总)=({hstr(res.count)}/{hstr(res.capacity)})"
    elif isinstance(res, Container):
        return f"(用/总)=({hstr(res.capacity - res.level)}/{hstr(res.capacity)})"

    raise ValueError(f"{res} 不是有效的 simpy 资源类型")


class CPU(Resource):
    def __init__(self, env: "SimEnv", num: int):
        super().__init__(env, capacity=num)

    @property
    def used_count(self) -> int:
        return self.count

    @property
    def free_count(self) -> int:
        cnt = self.capacity - self.used_count
        assert cnt >= 0
        return cnt

    @property
    def total_count(self) -> int:
        return self.capacity


class DataLink(Resource):
    """数据链路

    - 可用于限制设备同时发送/接受的数据数量
    - 不设限制时可以设置数量为一个比较大的值，比如 1000
    """

    def __init__(self, env: "SimEnv", num: int):
        super().__init__(env, capacity=num)

    @property
    def used_count(self) -> int:
        return self.count

    @property
    def free_count(self) -> int:
        cnt = self.capacity - self.used_count
        assert cnt >= 0
        return cnt

    @property
    def total_count(self) -> int:
        return self.capacity


class Memory(Container):
    """设备的内存"""

    def __init__(self, env: "SimEnv", size: int):
        super().__init__(env, init=size, capacity=size)

    @property
    def used_amount(self):
        return self.capacity - self.level

    @property
    def free_amount(self):
        return self.level

    @property
    def total_amount(self):
        return self.capacity


class Internet(Entity):
    delay: "defaultdict[DataSenderInternet, defaultdict[DataReceiverInternet, float]]"
    bandwidth: "defaultdict[DataSenderInternet, defaultdict[DataReceiverInternet, float]]"

    def __init__(self, env: "SimEnv"):
        super().__init__(env)
        self.delay = defaultdict(defaultdict)
        self.bandwidth = defaultdict(defaultdict)

    def set_conn_delay(self, src: "DataSenderInternet", dst: "DataReceiverInternet", delay: float):
        self.delay[src][dst] = delay

    def set_conn_bandwidth(self, src: "DataSenderInternet", dst: "DataReceiverInternet", bandwidth: float):
        self.bandwidth[src][dst] = bandwidth

    def get_conn_delay(self, src: "DataSenderInternet", dst: "DataReceiverInternet") -> float:
        assert dst in self.delay[src], f"必须为 {src} 和 {dst} 定义网络时延"
        return self.delay[src][dst]

    def get_conn_bandwidth(self, src: "DataSenderInternet", dst: "DataReceiverInternet") -> float:
        assert dst in self.bandwidth[src], f"必须为 {src} 和 {dst} 定义带宽"
        return self.bandwidth[src][dst]


class TransmittableData(Entity):
    trans_size: float
    trans_delay: float
    trans_delays: "List[float]"

    def __init__(self, env: "SimEnv", **kwargs):
        super().__init__(env, **kwargs)
        self.trans_size = 0.
        self.trans_delay = 0.
        self.trans_delays = []


class ComputableData(Entity):
    compute_size: float
    compute_delay: float
    compute_delays: "List[float]"
    computed_by: "List[DataProcessor]"

    def __init__(self, env: "SimEnv", **kwargs):
        super().__init__(env, **kwargs)
        self.compute_size = 0.
        self.compute_delay = 0.
        self.compute_delays = []
        self.computed_by = []


class ComputableDataEx(ComputableData):
    memory_size: float

    def __init__(self, env: "SimEnv"):
        super().__init__(env)
        self.memory_size = 128 * M


class DataReceiver(Entity):
    data_in_links: "DataLink"
    data_inbox: Store
    trans_queue: "List[float]"
    trans_bandwidths: "List[float]"
    trans_started_at: "List[float]"

    @property
    def trans_queue_span(self) -> float:
        """传输队列长度"""
        return sum(self.trans_queue) - sum([self.env.now - started_at for started_at in self.trans_started_at])

    @property
    def trans_queue_span_per_link(self) -> float:
        """传输队列平均长度"""
        return self.trans_queue_span / self.data_in_links.total_count

    @property
    def trans_available_in(self) -> float:
        """传输可用时刻的估计值"""
        if self.trans_queue_span == 0.:
            return 0.

        total_bandwidth = sum(self.trans_bandwidths)
        return self.trans_queue_span / total_bandwidth

    def __init__(self, env: "SimEnv"):
        super().__init__(env)
        self.data_in_links = DataLink(env, 1)
        self.data_inbox = Store(env)
        self.trans_queue = []
        self.trans_bandwidths = []
        self.trans_started_at = []


class DataReceiverInternet(DataReceiver):
    def __init__(self, env: "SimEnv"):
        super().__init__(env)


class DataReceiverPrivateNet(DataReceiver):
    data_in_bandwidth: float

    def __init__(self, env: "SimEnv"):
        super().__init__(env)
        self.data_in_bandwidth = 10 * M


class DataSender(Entity):
    data_outbox: Store
    data_out_links: Resource

    def __init__(self, env: "SimEnv"):
        super().__init__(env)
        self.data_outbox = Store(env)
        self.data_out_links = DataLink(env, 1)

    @abstractmethod
    def get_conn_delay(self, src: "DataSender", dst: "DataReceiver") -> float:
        raise NotImplementedError

    @abstractmethod
    def get_conn_bandwidth(self, src: "DataSender", dst: "DataReceiver") -> float:
        raise NotImplementedError

    def send_data(self, data: "TransmittableData", target: "DataReceiver"):
        self.trace(f'=> {data} 源={self} 目标={target} 传输量={hstr(data.trans_size)}')

        if self == target:
            self.log(f"<> {data} 本地回环")
            yield target.data_inbox.put(data)
        else:
            conn_delay = self.get_conn_delay(self, target)
            assert conn_delay is not None, '必须定义网络延迟'

            bandwidth = self.get_conn_bandwidth(self, target)
            assert bandwidth != 0. and bandwidth is not None, '必须定义网络带宽'

            trans_delay = conn_delay + data.trans_size / bandwidth
            target.trans_bandwidths.append(bandwidth)
            target.trans_queue.append(trans_delay)

            with self.data_out_links.request() as out_req:
                yield out_req
                self.trace(f'+ {data} 占用源传出链接 ')
                with target.data_in_links.request() as in_req:
                    yield in_req
                    self.trace(f'+ {data} 占用目标传入链接 ')

                    self.log(
                        f'=> {data} 传输 源={self} 目标={target} 传输量={hstr(data.trans_size)} 延迟={conn_delay}'
                        f' 带宽={hstr(bandwidth)} 预期完成于={self.env.now + trans_delay:.2f}')

                    yield self.env.timeout(trans_delay)
                    target.data_inbox.put(data)
                    data.trans_delay += trans_delay
                    data.trans_delays.append(trans_delay)

                    self.trace(f"<= {data} 传输 源={self} 目标={target}")

                self.trace(f"- {data} 释放目标传入链接")
                target.trans_queue.remove(trans_delay)
                target.trans_bandwidths.remove(bandwidth)

            self.trace(f"- {data} 释放源传出链接")

        self.trace(f'<= {data} 源={self} 目标={target}')


class DataSenderPrivateNet(DataSender):
    data_out_bandwidth: float

    def __init__(self, env: "SimEnv"):
        super().__init__(env)
        self.data_out_bandwidth = 10 * M

    def get_conn_delay(self, src: "DataSender", dst: "DataReceiver") -> float:
        return 0.

    def get_conn_bandwidth(self, src: "DataSenderPrivateNet", dst: "DataReceiverPrivateNet") -> float:
        return min(src.data_out_bandwidth, dst.data_in_bandwidth)


class DataSenderInternet(DataSender):
    env: "InternetSimEnv"

    def __init__(self, env: "InternetSimEnv"):
        super().__init__(env)

    def get_conn_delay(self, src: "DataSenderInternet", dst: "DataReceiverInternet") -> float:
        return self.env.net.get_conn_delay(src, dst)

    def get_conn_bandwidth(self, src: "DataSenderInternet", dst: "DataReceiverInternet") -> float:
        return self.env.net.get_conn_bandwidth(src, dst)


class DataProcessor(Entity):
    """具有CPU资源限制的数据处理设备"""
    computing_units: CPU
    compute_speed: float
    cpu_queue: "List[float]"
    cpu_started_at: "List[float]"

    @property
    def cpu_queue_span(self) -> float:
        """待处理任务的总计算时长"""
        return sum(self.cpu_queue) - sum([self.env.now - started_at for started_at in self.cpu_started_at])

    @property
    def cpu_queue_span_per_core(self) -> float:
        """待处理任务的预期平均计算时长（按核心数平均）"""
        return self.cpu_queue_span / self.computing_units.total_count

    def __init__(self, env: "SimEnv"):
        super().__init__(env)
        self.computing_units = CPU(env, 1)
        self.compute_speed = 1 * G
        self.cpu_queue = []
        self.cpu_started_at = []
        """CPU 启动的时刻"""

    @property
    def cpu_count(self):
        return self.computing_units.total_count

    @property
    def cpu_free(self):
        return self.computing_units.free_count

    @property
    def cpu_available_in(self) -> float:
        """CPU 相对于当前的时刻的预期的时间

        eg. 返回 512，表示在时刻 env.now + 512 时，该设备预期（可能）有可供使用的 CPU

        - 当设备尚有空闲 CPU 时返回 0，表示立刻可用。
        - 当设备的 CPU 全部被占用时，返回 **平均** 等待时间。"""

        if self.computing_units.free_count > 0:
            return 0.
        return self.cpu_queue_span / self.computing_units.total_count

    def notify_compute_task(self, data: "ComputableData"):
        compute_delay = data.compute_size / self.compute_speed
        self.cpu_queue.append(compute_delay)

    def compute(self, data: "ComputableData"):
        self.trace(f"=> {data}")
        assert self.cpu_queue_span > 0., "在 compute 前必须调用 notify_compute_task 以更新 CPU 队列长度"

        compute_delay = data.compute_size / self.compute_speed

        with self.computing_units.request() as req:
            yield req
            self.trace(f"+ {data} 占用处理单元")

            job_started_at = self.env.now
            self.cpu_started_at.append(job_started_at)

            data.computed_by = self
            yield self.env.timeout(compute_delay)
            self.log(f"✓✓ {data} 运算")
            data.compute_delay += compute_delay
            data.compute_delays.append(compute_delay)

        self.cpu_queue.remove(compute_delay)
        self.cpu_started_at.remove(job_started_at)
        assert self.cpu_queue_span >= 0., "CPU 队列时长必须大于等于0；在 compute 前必须调用 notify_compute_task"
        self.trace(f'- {data} 释放处理单元')
        self.trace(f'<= {data}')


class DataProcessorEx(DataProcessor):
    """具有CPU和内存限制的数据处理设备"""
    memory: Memory
    cost: float
    price_factor: float

    def __init__(self, env: "SimEnv"):
        super().__init__(env)
        self.memory = Memory(env, 1024)
        self.cost = 0.
        self.price_factor = 0.

    @property
    def memory_size(self):
        return self.memory.total_amount

    @property
    def memory_free(self):
        return self.memory.free_amount

    def compute(self, data: "ComputableDataEx"):
        if data.memory_size > self.memory_size:
            raise Warning('该设备无法满足内存需求')

        self.trace(f"=> {data}")

        compute_delay = data.compute_size / self.compute_speed

        with self.computing_units.request() as req:
            yield req
            self.trace(f"+ {data} 占用处理单元 {stats(self.computing_units)} 预期空闲于={self.cpu_available_in:.2f}")

            yield self.memory.get(data.memory_size)
            self.trace(f"+ {data} 占用内存 {stats(self.memory)}")

            job_started_at = self.env.now
            self.cpu_started_at.append(job_started_at)

            cost = compute_delay * self.price_factor
            self.cost += cost
            self.log(f"=> {data} 运算 预计耗时={compute_delay:.2f} 价格={cost:.2f}"
                     f" 预计完成于={self.env.now + compute_delay:.2f} 设备预期可用倒计时={self.cpu_available_in:.2f}")
            yield self.env.timeout(compute_delay)

            data.computed_by = self
            data.compute_delay += compute_delay
            data.compute_delays.append(compute_delay)
            self.trace(f"✓✓ {data} 运算 耗时={compute_delay:.2f} 价格={cost:.2f}")

            yield self.memory.put(data.memory_size)
            self.trace(f'- {data} 释放内存 {stats(self.memory)}')

        self.cpu_queue.remove(compute_delay)
        self.cpu_started_at.remove(job_started_at)
        assert self.cpu_queue_span >= 0., "CPU 队列时长必须大于等于0；在 compute 前必须调用 notify_compute_task"

        self.trace(f'- {data} 释放处理单元 {stats(self.computing_units)}')
        self.trace(f'<= {data}')


class InternetSimEnv(SimEnv):
    net: "Internet"

    def __init__(self):
        super().__init__()
        self.net = Internet(self)
