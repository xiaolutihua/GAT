import os
import random
from typing import List

import matplotlib.gridspec
import numpy as np
import pydotplus
from matplotlib import pyplot as plt

from common.id_object import IDObject
from utils import pickle_load, pickle_dump, get_gv_root, get_all_gv_files_in_data_dir


class ApplicationNode(IDObject):
    app: "ApplicationDAG"

    pre_nodes: "List[ApplicationNode]"
    "前驱任务节点"

    next_nodes: "List[ApplicationNode]"
    "后继任务节点"

    memory_size: int
    "内存需求"

    compute_size: int
    "计算量"

    program_size: int
    "程序大小"

    result_size: int
    "运行结果大小"

    @property
    def id_prefix(self) -> str:
        return f"{self.app.id}_Node"

    def __init__(self, app: "ApplicationDAG"):
        self.app = app
        self.app.nodes.append(self)
        super().__init__()
        self.pre_nodes = []
        self.next_nodes = []
        self.memory_size = 128
        self.compute_size = 256
        self.program_size = 64
        self.result_size = 32


class ApplicationDAG(IDObject):
    filepath: str
    nodes: "List[ApplicationNode]"

    @property
    def id_prefix(self) -> str:
        return 'App'

    @property
    def total_compute_size(self) -> float:
        return sum([node.compute_size for node in self.nodes])

    def __init__(self):
        super().__init__()
        self.filepath = ''
        self.nodes = []

    @staticmethod
    def load(filepath: str) -> "ApplicationDAG":
        app = ApplicationDAG()
        app.filepath = filepath

        gv_document: pydotplus.Dot = pydotplus.graphviz.graph_from_dot_file(filepath)
        gv_nodes = gv_document.get_nodes()
        gv_edges = gv_document.get_edges()

        key_to_node = '_task_graph_node'

        gv_node_count = 0
        for gv_node in gv_nodes:
            gv_node_count = gv_node_count + 1
            gv_attributes: dict = gv_node.get_attributes()

            node = ApplicationNode(app)
            node.compute_size = float(gv_attributes['compute_size'].strip('"'))
            node.program_size = float(gv_attributes['trans_size'].strip('"'))
            node.result_size = float(gv_attributes['result_size'].strip('"'))
            node.memory_size = float(gv_attributes['ram'].strip('"'))

            gv_node.set(key_to_node, node)

        for gv_edge in gv_edges:
            gv_src_node_name = gv_edge.get_source()
            gv_dst_node_name = gv_edge.get_destination()

            src_node: 'ApplicationNode' = gv_document.get_node(gv_src_node_name)[0].get(key_to_node)
            dst_node: 'ApplicationNode' = gv_document.get_node(gv_dst_node_name)[0].get(key_to_node)
            src_node.next_nodes.append(dst_node)
            dst_node.pre_nodes.append(src_node)

        return app


APP_CACHE_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/apps.pickle')


def get_apps_from_dir(data_dir: str = None) -> "List[ApplicationDAG]":
    if data_dir is None:
        data_dir = get_gv_root()

    gv_files = get_all_gv_files_in_data_dir(data_dir)

    apps = []
    for file in gv_files:
        app = ApplicationDAG.load(file)
        apps.append(app)

    pickle_dump(apps, APP_CACHE_FILE)
    return apps


def get_apps(use_cache: bool = True) -> "List[ApplicationDAG]":
    if use_cache and os.path.exists(APP_CACHE_FILE):
        # noinspection PyBroadException
        try:
            apps = pickle_load(APP_CACHE_FILE)
        except:
            print(f"加载 APP 缓存失败 APP_CACHE_FILE={APP_CACHE_FILE}，正重新建立缓存，可能需要较长时间。")
            apps = None
        if apps is not None and isinstance(apps, list) and isinstance(apps[0], ApplicationDAG):
            return apps
    return get_apps_from_dir()


APPS = get_apps()


def get_random_app():
    return random.choice(APPS)


def summary_of_app():
    apps = get_apps()
    stats = [[node.compute_size, node.program_size, node.memory_size, node.result_size]
             for app in apps for node in app.nodes]
    stats = np.array(stats).transpose().tolist()
    stat_labels = ['compute size', 'program size', 'memory size', 'result size']

    app_total_compute_size = [app.total_compute_size for app in apps]
    app_mean_memory_size = [np.mean([node.memory_size for node in app.nodes]) for app in apps]

    fig = plt.figure(tight_layout=True)
    gs = matplotlib.gridspec.GridSpec(2, 4)
    ax = fig.add_subplot(gs[0, 0:3])
    ax.scatter(app_total_compute_size, app_mean_memory_size)
    ax.set_xlabel('total compute size')
    ax.set_ylabel('mean memory size')

    ax = fig.add_subplot(gs[0, 3])
    ax.boxplot(app_total_compute_size, labels=['total compute size'])
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        ax.ticklabel_format(axis='y', style='sci', scilimits=[0, 0], useMathText=True)
        ax.boxplot(stats[i], labels=[stat_labels[i]])
        print(stat_labels[i],
              'min', np.min(stats[i]),
              '.25', np.percentile(stats[i], 25),
              'mean', np.mean(stats[i]),
              '.75', np.percentile(stats[i], 75),
              'max', np.max(stats[i]))
    plt.show()


def main():
    summary_of_app()


if __name__ == '__main__':
    main()
