import os
import uuid
from abc import abstractmethod
from typing import TypedDict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete, Dict, Box

from binbin.builder import get_basic_env, get_my_env
from binbin.devices import IoT
from binbin.env import BinbinSimEnv
from binbin.task import Task
from common.id_object import reset_id
from common.logging import LoggingContext, LOGGING_LEVEL_NO_LOG
from utils import pickle_dump, sparse_matrix_2_adj
from gat.graph_attention import Coder
from math import exp

SUVIVIE_REWARD = False
PUNISH_REWARD = False
ALPHA = 10


class EnvState(TypedDict):
    now: float


class DeviceState(TypedDict):
    total_cost: float
    cost: "List[float]"
    cpu_available_in: "List[float]"
    cpu_queue_span: "List[float]"
    cpu_queue_span_per_core: "List[float]"
    trans_available_in: "List[float]"
    cpu_count: "List[int]"
    cpu_free: "List[int]"
    cpu_speed: "List[float]"
    memory_size: "List[int]"
    memory_free: "List[int]"


class TaskState(TypedDict):
    offload_mask: "List[bool]"
    trans_size: float
    compute_size: float
    tolerance: float


class TaskpoolState:
    def __init__(self, tasks: "List[Task]"):
        self.tasks = []
        self.adj = []
        self.tasks.append([0, 0, 0, 0, 0])
        for task in tasks:
            self.tasks.append(
                [task.compute_size / 1.3e12,
                 task.program_size / 5e7,
                 task.memory_size / 1.1e6,
                 task.result_size / 1e5,
                 task.tolerance / 5e3]
            )

    def add_connection(self, start, end):
        self.adj.append([start, end])


Observation = Union[EnvState, DeviceState, TaskState]


def get_device_observation(sim: "BinbinSimEnv"):
    cpu_available_in = [dev.cpu_available_in for dev in sim.all_devices]
    """预期 CPU 可用时刻"""  # Box(0., np.inf, (all_dev_len,))
    trans_available_in = [dev.trans_available_in for dev in sim.all_devices]
    """预期网络传输可用时刻"""  # Box(0., np.inf, (all_dev_len,))
    memory_free = [dev.memory_free / 2 ** 30 for dev in sim.all_devices]
    """可用内存大小"""  # Box(0., np.inf, (all_dev_len,))
    device_state = [
        *cpu_available_in,
        *memory_free,
        *trans_available_in
    ]
    return device_state


def get_device_state(sim: "BinbinSimEnv") -> "DeviceState":
    total_cost = sum([dev.cost for dev in sim.shared_devices])
    """外移计算总费用"""  # Box(0., np.inf, ())
    cost = [dev.cost for dev in sim.shared_devices]
    """外移计算费用"""  # Box(0., np.inf, ())
    cpu_available_in = [dev.cpu_available_in for dev in sim.all_devices]
    """预期 CPU 可用时刻"""  # Box(0., np.inf, (all_dev_len,))
    cpu_queue_span = [dev.cpu_queue_span for dev in sim.all_devices]
    """CPU 任务队列长度"""  # Box(0., np.inf, (all_dev_len,))
    cpu_queue_span_per_core = [dev.cpu_queue_span_per_core for dev in sim.all_devices]
    """CPU 任务队列平均长度（按核心数平均）"""  # Box(0., np.inf, (all_dev_len,))
    trans_available_in = [dev.trans_available_in for dev in sim.all_devices]
    """预期网络传输可用时刻"""  # Box(0., np.inf, (all_dev_len,))
    cpu_count = [dev.cpu_count for dev in sim.all_devices]
    """CPU数量"""  # Box(0., np.inf, (all_dev_len,))
    cpu_free = [dev.cpu_free for dev in sim.all_devices]
    """可用的CPU数量"""  # Box(0., np.inf, (all_dev_len,))
    cpu_speed = [dev.compute_speed for dev in sim.all_devices]
    """CPU速度"""  # Box(0., np.inf, (all_dev_len,))
    memory_size = [dev.memory_size for dev in sim.all_devices]
    """内存大小"""  # Box(0., np.inf, (all_dev_len,))
    memory_free = [dev.memory_free for dev in sim.all_devices]
    """可用内存大小"""  # Box(0., np.inf, (all_dev_len,))

    return {
        "total_cost": total_cost,
        "cost": cost,
        "cpu_available_in": cpu_available_in,
        "cpu_queue_span": cpu_queue_span,
        "cpu_queue_span_per_core": cpu_queue_span_per_core,
        "trans_available_in": trans_available_in,
        "cpu_count": cpu_count,
        "cpu_free": cpu_free,
        "cpu_speed": cpu_speed,
        "memory_size": memory_size,
        "memory_free": memory_free
    }


def cast_to_gv(tasks: "List[Task]", cast_flag=False):
    """
    step1: 先获取head 节点
    step2: 然后一次往后探索所有的节点 , 加入到节点的网路中
    """
    taskpool_state = TaskpoolState(tasks)
    head_tasks = [task for task in tasks if
                  all([True if pre_task.allocated_to is not None else False for pre_task in task.pre_tasks])]
    # 头结点不代表就能够进行分配, 目前还没有对不能分配的头结点进行区分
    head_tasks = set(head_tasks)
    for task in head_tasks:
        idx = tasks.index(task)
        taskpool_state.add_connection(0, idx + 1)
    # 将现在已有的情况进行采集, 转化为gv 图
    while len(head_tasks) != 0:
        child_tasks = set()
        # 将head节点添加到 gv 文件
        for task in head_tasks:
            for child_task in task.next_tasks:
                child_tasks.add(child_task)
                tasks.index(task)
                pre_idx = tasks.index(task)
                child_idx = tasks.index(child_task)
                taskpool_state.add_connection(pre_idx + 1, child_idx + 1)
                # print(f"{pre_idx} -> {child_idx}")
                # 获取task和child_task的下标, 将其添加到gv图中
        assert all([task in tasks for task in child_tasks])
        head_tasks = child_tasks
    if cast_flag:
        pickle_dump(taskpool_state, f"{os.path.dirname(__file__)}/../../graph_datas/{uuid.uuid1()}.graph")
    return taskpool_state


def get_task_pool_state(env: "GymEnv"):
    """
    获取当前还没有分配任务的状态
    """
    tasks = env.sim.unallocated_tasks
    # 转换为gv文件
    return cast_to_gv(tasks)


"""
获取当前任务的状态
"""


def get_task_state(env: "GymEnv") -> "TaskState":
    offload_mask = [False] * env.all_dev_len
    for i in range(env.all_dev_len):
        device = env.sim.all_devices[i]
        if isinstance(device, IoT):
            if device == env.task.generated_by:
                offload_mask[i] = True
        else:
            offload_mask[i] = True

    return {
        "offload_mask": offload_mask,
        "compute_size": env.task.compute_size,
        "trans_size": env.task.trans_size,
        "tolerance": env.task.tolerance
    }


class GymEnv(gym.Env):
    sim: "BinbinSimEnv"
    all_dev_len: int
    shared_dev_len: int
    action_space_len: int
    task: "Optional[Task]"
    step_count: int
    encoder: "Optional[Coder]"

    @property
    def ACTION_DIM(self):
        return self.action_space_len

    def __init__(self,
                 encoder_weight="tests/gat/encoder-checkpoint10.pt",
                 encoder_task_dims: int = 5,
                 encoder_task_state: bool = True,
                 iot_num: int = 6,
                 fog_num: int = 5,
                 cycle_reduce_rate: float = 0.6,
                 time_unit_fix=None,
                 iot_bandwidth=10,
                 task_nums=20,
                 ) -> None:
        super().__init__()
        self.encoder_weight_path = encoder_weight
        self.encoder_task_state = encoder_task_state
        self.encoder_task_dims = encoder_task_dims
        self.device_dim = 3
        self.n_node_feature = 5
        self.iot_num = iot_num
        self.fog_num = fog_num
        self.cycle_reduce_rate = cycle_reduce_rate
        self.time_unit_fix = time_unit_fix
        self.iot_bandwidth = iot_bandwidth
        self.task_nums=task_nums
        with LoggingContext(LOGGING_LEVEL_NO_LOG):
            self.reset()

        self.action_space = Discrete(self.action_space_len)
        # self.observation_space = Dict({
        #     "total_cost": Box(0., np.inf, ()),
        #     "cpu_available_in": Box(0., np.inf, (self.shared_dev_len,))
        # })
        self.observation_space = Dict(
            {
                "tasks": Box(0, np.inf, (encoder_task_dims,)),
                "devices": Box(0, np.inf, (self.device_dim * self.all_dev_len,)),
            }
        )
        print("准备初始化encoder")

    def only_mec_cost(self):
        # 首先获取所有终端设备的任务的计算量
        edge_only = 0
        for iot in self.sim.iots:
            edge_only += iot.app.total_compute_size / 2 ** 30 * 0.01 * (self.sim.app_cycle_lcm // iot.app_cycle)
        return edge_only

    def step(self, action):
        if action == 0:
            allocated_to = self.task.generated_by
            cost = 0
        elif 1 <= action <= len(self.sim.fogs):
            allocated_to = self.sim.fogs[action - 1]
            cost = self.task.compute_size / 2 ** 30 * 0.01
        else:
            allocated_to = self.sim.clouds[action - len(self.sim.fogs) - 1]
            cost = self.task.compute_size / 2 ** 30 * 0.02

        self.task.allocated_to = allocated_to
        self.task.allocator_notify()

        _, is_task_timeout, is_env_done = self.sim.run_until_allocation_requested_or_timeout_or_done()

        self.step_count += 1
        terminated = is_task_timeout
        truncated = is_env_done

        if SUVIVIE_REWARD:
            reward = 1 if not is_task_timeout else 0  # TODO 存活奖励1+cost奖励
        else:
            reward = 0
        # reward += exp(-cost)
        reward += -cost
        if terminated and PUNISH_REWARD:
            reward += -10000
        if self.encoder_task_state:
            obs, info = self._get_observation_gat(), self._get_obs()  # TODO 完成observation的设计
        else:
            obs, info = self._get_observation_allocate_task(), self._get_obs()
        # obs, info = self._get_obs(), self._get_info()
        # print("打印观测状态", obs)
        # 获取任务池的状态
        # 获取设备的状态 , 拼接
        info["cost"] = cost
        self.task = self._get_next_task()
        return obs, reward, terminated, truncated, info

    def _get_next_task(self) -> "Optional[Task]":
        if len(self.sim.allocation_queue) != 0:
            return self.sim.allocation_queue.pop(0)
        else:
            return None

    """从simulator获取状态, 然后经过encoder处理"""

    def _get_observation_gat(self):
        taskpool_state = get_task_pool_state(self)
        x = torch.tensor(taskpool_state.tasks, dtype=torch.float32)
        # 将 spares_matrix 转化为 adjust_matrix

        if len(taskpool_state.adj) > 0:
            adj = sparse_matrix_2_adj(taskpool_state.adj)
            task_states = self.encoder.forward(x, adj)
            task_state = task_states[0].tolist()
        else:
            task_state = [0.] * 5

        device_state = get_device_observation(self.sim)
        return [
            *task_state,
            *device_state
        ]

    def _get_observation_allocate_task(self):
        task_state = [self.task.compute_size,
                      self.task.result_size,
                      self.task.program_size,
                      self.task.memory_size,
                      self.task.tolerance]
        device_states = get_device_observation(self.sim)
        return [
            *task_state,
            *device_states
        ]

    def _get_obs(self) -> "Observation":
        device_state = get_device_state(self.sim)
        task_state = get_task_state(self)
        taskpool_state = get_task_pool_state(self)

        observation = {
            "now": self.sim.now,
            "taskpool": taskpool_state,
            **device_state,
            **task_state
        }
        return observation  # type: ignore

    class Info(TypedDict):
        now: float
        step_count: int
        task: "Optional[Task]"

    def _get_info(self) -> "Info":
        return {
            "now": f"{self.sim.now:.2f}",
            "step_count": self.step_count,
            "task": f"{self.task} allocated_to={self.task.allocated_to}" if self.task is not None else "(No task)"
        }

    def reset(self, **kwargs) -> "Tuple(Any, Info)":
        # print("环境准备reset")
        reset_id()
        self.sim = self.build_sim_env()
        self.all_dev_len = len(self.sim.all_devices)
        if self.encoder_task_state:
            self.STATE_DIM = self.encoder_task_dims + self.device_dim * self.all_dev_len
        else:
            self.STATE_DIM = self.n_node_feature + self.device_dim * self.all_dev_len
        self.shared_dev_len = len(self.sim.shared_devices)
        self.action_space_len = self.shared_dev_len + 1
        self.step_count = 0
        self.sim.start()
        self.sim.run_until_allocation_requested_or_timeout_or_done()
        self.task = self._get_next_task()
        # 用来控制状态输出的
        if self.encoder_task_state:
            self.encoder = Coder(self.n_node_feature, 10, self.encoder_task_dims, 0.2, 0.2, 5, 3, False)
            self.encoder.load_state_dict(torch.load(self.encoder_weight_path))
            obs, info = self._get_observation_gat(), self._get_obs()  # info 是greedy的状态
        else:
            obs, info = self._get_observation_allocate_task(), self._get_obs()
            self.encoder = None
        # obs, info = self._get_obs(), self._get_info()
        return obs, info

    @abstractmethod
    def build_sim_env(self):
        # env = get_basic_env()
        env, time_unit, average_reduce_rate, total_app_nums = get_my_env(
            iot_num=self.iot_num,
            fog_num=self.fog_num,
            cycle_reduce_rate=self.cycle_reduce_rate,
            time_unit_fix=self.time_unit_fix,
            iot_bandwidth=self.iot_bandwidth,
            task_nums=self.task_nums
        )
        self.time_unit = time_unit
        self.average_reduce_rate = average_reduce_rate
        self.total_app_nums = total_app_nums
        env.generated_app_nums_target = total_app_nums
        print(f"time_unit:{time_unit}, average_reduce_rate:{average_reduce_rate}, total_app_nums:{total_app_nums}")
        return env

    def render(self, mode="human"):
        raise NotImplementedError
