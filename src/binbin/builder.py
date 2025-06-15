import os.path
import random

from binbin.application import ApplicationDAG, get_random_app
from binbin.env import BinbinSimEnv
from binbin.preset import make_iot_s, make_iot_m, make_iot_l, make_cloud_m, make_fog_s, make_fog_m
from constants import hstr


# 设计的比例：三七开（三分自算，七分卸载）
# 以下规格均省略单位
#
# 云计算 参考阿里云       32~64CPU  2.7~3.7GHz    32G
# 雾计算 家用PC算力       8~16CPU   2.9~4.2GHz    16~32G
# IOT   参考树莓派       1~4CPU    0.7~1.5GHz    0.256G~2G（为了规避 OOM 必须大于最大子任务的内存需求）
#
# 阿里云规格 <https://www.alibabacloud.com/help/zh/ecs/user-guide/compute-optimized-instance-families>
# 阿里云价格 <https://developer.aliyun.com/article/721836>
# 树莓派    <https://shumeipai.nxez.com/raspberry-pi-version-compare>
#
# 网络（带宽/时延）
# inter-iot    10M     0.50  应该不会用到
# iot-outside  10M     0.30
# fog-cloud    100M    0.05
# inter-fog    100M    0.01
# inter-cloud  1000M   0.01

# 测试数据信息
# compute_size 1e12   1000G (1T)  mean 228G max 1426G
# program_size 1e6    1M
# memory_size  1e7    10M         max 52M
# result_size  1e4    result 传输可以忽略不计，因为比 program 小两个数量级别
#

# hyb 修改IoT个数
def get_my_env(iot_num: int,
               fog_num: int,
               cycle_reduce_rate: float,
               time_unit_fix=None,
               iot_bandwidth=10,
               task_nums=20,
               ):
    # 必须先创建IoT 设备
    env = BinbinSimEnv(iot_bandwidth)
    for i in range(iot_num):
        app1 = ApplicationDAG.load(f'{os.path.dirname(__file__)}/data/task-graph/{task_nums}/random_{task_nums}_{i}.gv')
        iot = make_iot_s(env, app1)
    for i in range(fog_num):
        make_fog_m(env)
    # 周期处理
    max_speed = max([node.compute_speed for node in env.all_devices])
    print(f"max_speed: {max_speed}")

    if time_unit_fix is not None:
        print("使用固定的time_unit")
        average_reduce_rate = 0
        app_num = 0
        for iot in env.iots:
            iot_cycle, local_cycle, min_cycle = iot.set_cycle(
                max_speed,
                time_unit_fix,
                cycle_reduce_rate
            )
            app_num += iot_cycle / time_unit_fix
            average_reduce_rate += 1 - (iot_cycle - min_cycle) / (local_cycle - min_cycle)
        return env, time_unit_fix, average_reduce_rate/len(env.iots), app_num
    else:
        # 在某个搜索区间之间搜索合适的time_unit找到最小的实现
        time_unit = 100

        app_nums = []
        time_units = []
        average_reduce_rates = []
        while time_unit <= 3000:
            app_count = 0  # 在lcm的时间内存在多少个应用
            flag = True
            average_reduce_rate = 0
            for iot in env.iots:
                iot_cycle, local_cycle, min_cycle = iot.set_cycle(
                    max_speed,
                    time_unit,
                    cycle_reduce_rate
                )
                average_reduce_rate += 1 - (iot_cycle - min_cycle) / (local_cycle - min_cycle)
                app_count += iot_cycle / time_unit
                if iot_cycle >= local_cycle:
                    flag = False
                    break
            if flag:
                app_nums.append(app_count)
                time_units.append(time_unit)
                average_reduce_rates.append(average_reduce_rate / len(env.iots))

            time_unit += 50
        # 设置APP数量最小的time_unit
        print(f"app_nums:{app_nums}, time_units:{time_units}, average_reduce_rates:{average_reduce_rates}")
        indexs = []
        for i,v in enumerate(app_nums):
            if v == min(app_nums):
                indexs.append(i)
        reduce_rates = [average_reduce_rates[i] for i in indexs]
        index = indexs[reduce_rates.index(max(reduce_rates))]

        average_reduce_rate = 0
        for iot in env.iots:
            iot_cycle, local_cycle, min_cycle = iot.set_cycle(
                max_speed,
                time_units[index],
                cycle_reduce_rate
            )
            average_reduce_rate += 1 - (iot_cycle - min_cycle) / (local_cycle - min_cycle)
        return env, time_units[index], average_reduce_rate/len(env.iots), app_nums[index]


def get_basic_env() -> "BinbinSimEnv":
    """用于手动调试的基本环境

    - 忍耐极限设置为512且使用 app1，少于或等于两个 ioT 设备大概成成功，小概率失败
    """
    env = BinbinSimEnv()

    app1 = ApplicationDAG.load(f'{os.path.dirname(__file__)}/data/task-graph/simple/g1.gv')

    for i in range(2):
        iot = make_iot_s(env)
        iot.app = random.choice([app1])

    make_fog_s(env)
    make_cloud_m(env)

    return env


def get_random_heavy_env():
    env = BinbinSimEnv()

    for _ in range(16):
        iot = make_iot_s(env)
        iot.app = get_random_app()
        iot.app_cycle = 3200

    make_fog_m(env)
    make_cloud_m(env)
    return env


def get_env_1() -> "BinbinSimEnv":
    env = BinbinSimEnv()
    make_iot_s(env)
    return env


def check_env(env: "BinbinSimEnv"):
    """快速检查环境设置是否合理

    主要是检查环境的算力要求，因为网络传输几乎可以忽略不计
    """

    capital_t = env.app_cycle_lcm
    total_compute_size_in_capital_t = 0.
    for iot in env.iots:
        task_group_compute_size = sum([node.compute_size for node in iot.app.nodes])
        task_group_repeats = capital_t // iot.app_cycle
        total_compute_size_in_capital_t += task_group_compute_size * task_group_repeats

    total_iot_compute_speed = sum([iot.compute_speed * iot.cpu_count for iot in env.iots])  # 所有 IoT 设备的满载算力
    total_shared_compute_speed = sum([dev.compute_speed * dev.cpu_count for dev in env.shared_devices])  # 所有云/雾设备满载算力

    for offload_rate in range(0, 101, 10):
        average_speed = (total_iot_compute_speed * (offload_rate / 100) +
                         total_shared_compute_speed * (1 - offload_rate / 100))
        consumed_time = total_compute_size_in_capital_t / average_speed
        print(f'卸载率={offload_rate}% 满载计算理论速度={hstr(average_speed)} 满载计算理论时间={consumed_time}')


def main():
    env = get_random_heavy_env()
    check_env(env)


if __name__ == '__main__':
    main()
