import numpy as np

from binbin.analysis import Analyzer
from binbin.builder import get_random_heavy_env, get_my_env
from binbin.env import BinbinSimEnv
from binbin.gym_env import GymEnv, Observation
from common.logging import set_logging_level


class GreedyEnv(GymEnv):
    def build_sim_env(self) -> "BinbinSimEnv":
        # env = get_random_heavy_env()
        env,_,_,_ = get_my_env(iot_num=6, fog_num=5, cycle_reduce_rate=0.3)
        return env


def fastest_greedy(obs: "Observation") -> int:
    trans_delays = obs['trans_available_in']
    compute_speeds = np.array(obs['cpu_speed'])
    compute_delays = obs['cpu_queue_span_per_core'] + obs['compute_size'] / compute_speeds
    delays = trans_delays + compute_delays
    delays = delays[np.array(obs['offload_mask'])]
    delays = [delay if delay <= obs['tolerance'] else float('inf') for delay in delays]
    action = np.argmin(delays)
    return action


def run_greedy():
    env = GreedyEnv()
    analyzer = Analyzer()
    set_logging_level(0)

    observation, info = env.reset()
    analyzer.add_observation(info)
    cost = 0
    while True:
        action = fastest_greedy(info)
        observation, reward, terminated, truncated, info = env.step(action)
        cost += info["cost"]
        analyzer.add_observation(info)
        print("<<", f'action={action}', observation, reward, terminated, truncated, info)
        if terminated or truncated:
            print('!!', f"超时中止={terminated}", f"周期完成={truncated}")
            observation, info = env.reset()
            break
    print("贪心算法的cost: ", cost)
    analyzer.show()


if __name__ == '__main__':
    run_greedy()
