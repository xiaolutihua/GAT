from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING, Optional, Dict

if TYPE_CHECKING:
    from binbin.devices import DistributedComputingDevice, IoT
    from binbin.env import BinbinSimEnv

from simpy import Event

from binbin.application import ApplicationNode, ApplicationDAG
from common.data import TransmittableData, ComputableDataEx
from common.entity import Entity


@dataclass
class TreeNode:
    task: "Optional[Task]" = None
    memory_size: float = 0
    compute_size: float = 0
    program_size: float = 0
    result_size: float = 0
    next_nodes: "List[TreeNode]" = field(default_factory=list)

    def __str__(self):
        return self.task.id if self.task is not None else 'Virtual'

    @staticmethod
    def from_task(task: "Task", node_by_task: "Dict[Task, TreeNode]" = None) -> "TreeNode":
        if node_by_task is None:
            node_by_task = dict()

        if task not in node_by_task:
            node = TreeNode(
                task=task,
                memory_size=task.memory_size,
                compute_size=task.compute_size,
                program_size=task.program_size,
                result_size=task.result_size
            )
            node_by_task[task] = node

        node = node_by_task[task]
        for sub_task in task.next_tasks:
            node.next_nodes.append(TreeNode.from_task(sub_task, node_by_task))
        return node


class TaskGroup(Entity):
    env: "BinbinSimEnv"
    app: "ApplicationDAG"
    tasks: "List[Task]"
    task_group_completed: Event
    generated_at: float
    cycle: "Optional[float]"
    terminated_at: "Optional[float]"

    @property
    def id_prefix(self) -> str:
        return 'TG'

    @property
    def tolerance(self) -> float:
        return max(self.terminated_at - self.env.now, 0.)

    def __init__(self, env: "BinbinSimEnv", app: "ApplicationDAG"):
        super().__init__(env)
        self.app = app
        self.env.task_groups.append(self)
        self.tasks = []
        self.generated_at = env.now
        self.cycle = None
        self.terminated_at = None

    def add_task(self, task: "Task"):
        self.tasks.append(task)

    @property
    def task_group_completed_event(self) -> "Event":
        return self.env.all_of([task.completed for task in self.tasks])

    @property
    def is_task_group_completed(self) -> bool:
        return self.task_group_completed_event.triggered


class Task(ComputableDataEx, TransmittableData, Entity):
    node: "ApplicationNode"
    group: "TaskGroup"
    completed: Event
    allocator_notified_event: "Event"
    generated_by: "IoT"
    allocated_to: "Optional[DistributedComputingDevice]"

    @property
    def id_prefix(self) -> str:
        return self.group.id

    def __init__(self, env: "BinbinSimEnv", node: "ApplicationNode", group: "TaskGroup", iot: "IoT"):
        self.node = node
        self.group = group
        super().__init__(env)
        self.pre_tasks = []
        self.next_tasks = []
        self.memory_size = node.memory_size
        self.compute_size = node.compute_size
        self.program_size = node.program_size
        self.result_size = node.result_size
        self.completed = env.event()
        self.group.add_task(self)
        self.allocated_to = None
        self.allocator_notified_event = env.event()
        self.generated_by = iot

    def allocator_notify(self):
        if self.allocated_to is not None:
            self.allocated_to.notify_compute_task(self)
        self.allocator_notified_event.succeed()
        self.allocator_notified_event = self.env.event()

    @property
    def tolerance(self) -> float:
        return self.group.tolerance

    @property
    def is_completed(self) -> bool:
        return self.completed.triggered

    @property
    def is_allocated(self) -> bool:
        return self.allocated_to is not None

    @property
    def all_pre_tasks_completed_event(self) -> "Event":
        return self.env.all_of([task.completed for task in self.pre_tasks])

    @property
    def any_pre_tasks_completed_event(self) -> "Event":
        return self.env.any_of([task.completed for task in self.pre_tasks if not task.is_completed])

    @property
    def are_all_pre_tasks_completed(self) -> bool:
        return self.all_pre_tasks_completed_event.triggered

    @property
    def are_all_pre_tasks_allocated(self) -> bool:
        return all([task.is_allocated for task in self.pre_tasks])

    @property
    def must_be_allocated(self) -> bool:
        return self.are_all_pre_tasks_completed
