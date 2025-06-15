import random
from typing import List, Optional

from constants import M
from simpy import Event
from common.data import InternetSimEnv
from binbin.devices import IoT, Fog, Cloud, DistributedComputingDevice
from binbin.task import TaskGroup, Task, TreeNode
from utils import lcm


class BinbinSimEnv(InternetSimEnv):
    iots: "List[IoT]"
    fogs: "List[Fog]"
    clouds: "List[Cloud]"
    all_devices: "List[DistributedComputingDevice]"
    shared_devices: "List[DistributedComputingDevice]"
    task_groups: "List[TaskGroup]"
    allocation_queue: "List[Task]"
    env_done_event: "Event"
    task_group_timeout_event: "Event"
    allocation_requested_event: "Event"

    time_unit: int
    cycle_discount_rate: float
    generated_app_nums_target: "Optional[int]"
    generated_app_nums: int

    def __init__(self, iot_bandwidth=10):
        super().__init__()
        self.iots = []
        self.fogs = []
        self.clouds = []
        self.all_devices = []
        self.shared_devices = []
        self.task_groups = []
        self.allocation_queue = []
        self.env_done_event = self.event()
        self.task_group_timeout_event = self.event()
        self.allocation_requested_event = self.event()
        self.generated_app_nums_target = None
        self.generated_app_nums = 0
        self.iot_bandwidth = iot_bandwidth

    @property
    def app_cycle_lcm(self):
        ret = 1
        for iot in self.iots:
            assert iot.app_cycle == int(iot.app_cycle), "由于计算最小公倍数，必须采用整数的 APP 周期。"
            ret = lcm(int(iot.app_cycle), ret)
        # print("episode的时长", ret)
        return ret

    @property
    def total_cost(self) -> float:
        """所有计算设备的价格"""
        return sum([dev.cost for dev in self.shared_devices])

    def start(self):
        self.setup_net()
        super().start()
        self.process(self.watch_for_env_done())

    def watch_for_env_done(self):
        yield self.timeout(self.app_cycle_lcm)  # 等待epsiode_time完成
        if all([group.is_task_group_completed for group in self.task_groups if group.generated_at < self.now]):
            self.env_done_event.succeed()

    def setup_net(self):
        self.setup_net_for_group(self.iots, self.iots, self.iot_bandwidth * M, 0.5)
        self.setup_net_for_group(self.iots, self.fogs, self.iot_bandwidth * M, 0.3)
        self.setup_net_for_group(self.iots, self.clouds, self.iot_bandwidth * M, 0.3)
        self.setup_net_for_group(self.fogs, self.clouds, 100 * M, 0.05)
        self.setup_net_for_group(self.fogs, self.fogs, 100 * M, 0.01)
        self.setup_net_for_group(self.clouds, self.clouds, 1000 * M, 0.01)

    def setup_net_for_group(self,
                            group1: "List[DistributedComputingDevice]",
                            group2: "List[DistributedComputingDevice]",
                            bandwidth: float,
                            delay: float):
        for node1 in group1:
            for node2 in group2:
                if node1 == node2:
                    continue
                self.net.set_conn_delay(node1, node2, delay)
                self.net.set_conn_delay(node2, node1, delay)
                self.net.set_conn_bandwidth(node1, node2, bandwidth)
                self.net.set_conn_bandwidth(node2, node1, bandwidth)

    def run_until_allocation_requested_or_timeout_or_done(self) -> (bool, bool, bool):
        if len(self.allocation_queue) > 0:
            return True, False, False

        requested, timeout, done = self.allocation_requested_event, self.task_group_timeout_event, self.env_done_event
        event = self.any_of([requested, timeout, done])
        self.run(until=event)
        is_requested, is_task_timeout, is_done = requested.triggered, timeout.triggered, done.triggered
        return is_requested, is_task_timeout, is_done

    def request_allocate_task(self, task: "Task"):
        self.allocation_requested_event.succeed()
        self.allocation_requested_event = self.event()
        self.allocation_queue.append(task)

    def report_task_group_added(self, task_group: "TaskGroup"):
        self.process(self.watch_for_task_group_completed(task_group))
        self.process(self.watch_for_task_group_timeout(task_group))

    def watch_for_task_group_completed(self, task_group: "TaskGroup"):
        yield task_group.task_group_completed_event
        self.log(f"✓✓ 任务组 {task_group} 完成")

    def watch_for_task_group_timeout(self, task_group: "TaskGroup"):
        assert task_group.cycle is not None, f'任务组 {task_group} 必须设置周期'
        assert task_group.terminated_at is not None, f'任务组 {task_group} 必须设置强制终止时间点'

        yield self.timeout(task_group.tolerance)
        if not task_group.is_task_group_completed:
            self.log(f"!! 任务组超时 {task_group}")
            self.task_group_timeout_event.succeed(task_group)
            self.task_group_timeout_event = self.event()

    def random_allocate(self, task: "Task"):
        should_allocate = task.must_be_allocated or random.uniform(0, 1) > 0.6
        if should_allocate:
            targets = [task.generated_by]
            targets.extend(self.fogs)
            targets.extend(self.clouds)
            task.allocated_to = random.choice(targets)
        task.allocator_notify()

    @property
    def tasks(self) -> "List[Task]":
        """返回所有未完成的任务组中的所有子任务

        这些子任务可能已经完成，也可能未完成；可能已经被分配，也可能没有被分配。
        """

        tasks = []
        for group in self.task_groups:
            if group.is_task_group_completed:
                continue
            for task in group.tasks:
                tasks.append(task)
        return tasks

    @property
    def unallocated_tasks(self) -> "List[Task]":
        """返回所有未分配的子任务"""
        return list(filter(lambda task: not task.is_allocated, self.tasks))

    @property
    def task_pool(self) -> "TreeNode":
        tasks_in_pool: "List[Task]" = []

        for group in self.task_groups:
            if group.is_task_group_completed:
                continue

            for task in group.tasks:
                if task.are_all_pre_tasks_allocated and not task.is_allocated:
                    tasks_in_pool.append(task)

        node_by_task = dict()
        virtual_head = TreeNode()
        for task in tasks_in_pool:
            virtual_head.next_nodes.append(TreeNode.from_task(task, node_by_task))

        return virtual_head
