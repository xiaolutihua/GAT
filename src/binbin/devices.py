from typing import Optional, cast, TYPE_CHECKING

from simpy import Process

from binbin.application import ApplicationDAG
from constants import hstr

if TYPE_CHECKING:
    from binbin.env import BinbinSimEnv
from binbin.task import Task, TaskGroup
from common.data import DataProcessorEx, DataSenderInternet, DataReceiverInternet


class DistributedComputingDevice(DataSenderInternet, DataReceiverInternet, DataProcessorEx):
    env: "BinbinSimEnv"
    computing_action: "Optional[Process]"

    def __init__(self, env: "BinbinSimEnv"):
        super().__init__(env)
        self.computing_action = None

    def start(self):
        self.computing_action = self.env.process(self.distributed_computing())

    def distributed_computing(self):
        while True:
            data = yield self.data_inbox.get()
            task = cast("Task", data)
            self.trace(f"|> 收到 {task}")
            self.env.process(self.handle_inbox_task(task))

    def handle_inbox_task(self, task: "Task"):
        if task.is_completed:
            self.trace(f"|> 收到运行结果 {task}")
            return

        if task.is_allocated:
            if task.allocated_to != self:
                if task.generated_by == self:
                    yield self.env.process(self.offload_task(task))
                    return
                raise Warning('任务已被分配，但被发送到了错误的设备')
            else:
                yield self.env.process(self.process_task(task))
                return

        yield self.env.process(self.request_allocation(task))
        if task.allocated_to == self:
            yield self.env.process(self.process_task(task))
        else:
            yield self.env.process(self.offload_task(task))

    def request_pre_task_result(self, task, pre_task):
        if not pre_task.is_completed:
            self.trace(f"=> {task} 等待前置任务 {pre_task} 完成")
            yield pre_task.completed
            self.log(f"<= {task} 的前置任务 {pre_task} 完成")

        self.log(f"=> {task} 下载前置任务 {pre_task} 结果")
        pre_task.trans_size = pre_task.result_size
        pre_task_device = cast(DistributedComputingDevice, pre_task.allocated_to)
        yield self.env.process(pre_task_device.send_data(pre_task, self))
        self.trace(f"<= {task} 下载前置任务 {pre_task} 结果")

    def process_task(self, task):
        self.trace(f"=> {task} 处理")

        if len(task.pre_tasks) > 0:
            self.log(f"=> {task} 下载前置任务结果")
            yield self.env.all_of([
                self.env.process(self.request_pre_task_result(task, pre_task)) for pre_task in task.pre_tasks
            ])
            self.trace(f"✓✓ {task} 下载前置任务结果")

        yield self.env.process(self.compute(task))
        yield task.completed.succeed()
        self.log(f"✓✓ {task} 处理")
        self.trace(f"<= {task}")

    def request_allocation(self, task: "Task"):
        self.trace(f"=> {task}")

        while not task.is_allocated:
            if len(task.pre_tasks) > 0:
                self.trace(f"=> 在请求分配前等待所有前置任务完成 {task}")
                yield task.all_pre_tasks_completed_event
                self.trace(f"<= 所有前置任务完成 {task}")

            self.log(f"=> {self} 请求分配 {task}")
            self.env.request_allocate_task(task)

            yield task.allocator_notified_event
            self.trace(f"<= {self} 请求分配 {task}")

            if task.must_be_allocated and not task.is_allocated:
                raise Warning('任务必须被分配，但分配器未作出分配')

            if task.is_allocated:
                self.log(f"<= {task} 被分配至 {task.allocated_to}")
                return
            else:
                self.log(f"<= {task} 未被分配，等待下一次分配")

            yield task.any_pre_tasks_completed_event
        self.trace(f"<= {task}")

    def offload_task(self, task: "Task"):
        self.trace(f"=> {task} 卸载目标={task.allocated_to}")
        task.trans_size = task.program_size
        yield self.env.process(self.send_data(task, task.allocated_to))
        self.trace(f"<= {task} 卸载目标={task.allocated_to}")


class Fog(DistributedComputingDevice):
    def __init__(self, env: "BinbinSimEnv"):
        super().__init__(env)
        self.env.fogs.append(self)
        self.env.all_devices.append(self)
        self.env.shared_devices.append(self)


class Cloud(DistributedComputingDevice):
    def __init__(self, env: "BinbinSimEnv"):
        super().__init__(env)
        self.env.clouds.append(self)
        self.env.all_devices.append(self)
        self.env.shared_devices.append(self)


class IoT(DistributedComputingDevice):
    env: "BinbinSimEnv"
    app: "ApplicationDAG"
    app_cycle: float
    generate_task_action: "Optional[Process]"

    def __init__(self, env: "BinbinSimEnv", app: "ApplicationDAG"):
        super().__init__(env)
        self.env.iots.append(self)
        self.env.all_devices.append(self)
        self.app = app
        # self.app_cycle = 512
        # self.get_local_execute_time()
        self.generate_task_action = None

    # hyb cycle
    def set_cycle(self, max_compute_speed, time_unit: int, cycle_reduce_rate: float):
        assert self.app is not None, "app 为空"
        compute_size = 0
        for node in self.app.nodes:
            compute_size += node.compute_size
        import math
        app_cycle_local = compute_size / self.compute_speed
        app_cycle_faster = compute_size / max_compute_speed
        delta = app_cycle_local - app_cycle_faster
        self.app_cycle = app_cycle_faster + (1 - cycle_reduce_rate) * delta
        self.app_cycle = math.ceil(self.app_cycle / time_unit) * time_unit
        # print(f"本地计算: {app_cycle_local}, "
        #       f"最快的周期计算时间: {app_cycle_faster}, "
        #       f"设定的周期时间: {self.app_cycle}, "
        #       f"周期缩短率: {self.app_cycle/app_cycle_local}")
        return self.app_cycle, app_cycle_local, app_cycle_faster

    def start(self):
        super().start()
        self.generate_task_action = self.env.process(self.generate_task())

    def generate_task(self):
        while True:
            if self.env.generated_app_nums_target is not None:
                if self.env.generated_app_nums == self.env.generated_app_nums_target:
                    break
                elif self.env.generated_app_nums > self.env.generated_app_nums_target:
                    # 不可能进入这个分支
                    raise Exception("self.env.generated_app_nums > self.env.generated_app_nums_target 不可能成立。")
                self.env.generated_app_nums += 1

            task_group = self.generate_tasks_for_app()
            self.log(f"|> 任务组生成 {task_group} 周期={task_group.cycle}"
                     f" 总计算量:{hstr(task_group.app.total_compute_size)}"
                     f" 强制终止于={task_group.terminated_at}")
            for task in task_group.tasks:
                self.data_inbox.put(task)
            self.env.report_task_group_added(task_group)
            yield self.env.timeout(self.app_cycle)

    def generate_tasks_for_app(self) -> "TaskGroup":
        task_by_node = dict()
        group = TaskGroup(self.env, self.app)
        group.cycle = self.app_cycle
        group.terminated_at = self.env.now + self.app_cycle

        for node in self.app.nodes:
            task = Task(self.env, node, group, self)
            task_by_node[node] = task

        for node in self.app.nodes:
            task = task_by_node[node]
            for pre_node in node.pre_nodes:
                pre_task = task_by_node[pre_node]
                task.pre_tasks.append(pre_task)
            for next_node in node.next_nodes:
                next_task = task_by_node[next_node]
                task.next_tasks.append(next_task)
        return group
