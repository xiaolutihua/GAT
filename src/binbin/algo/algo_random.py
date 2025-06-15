from binbin.gym_env import GymEnv
from common.logging import LoggingContext, set_logging_level


def run_random():
    env = GymEnv()
    set_logging_level(2)

    observation, info = env.reset()
    print("!!", observation, info)

    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print("<<", observation, reward, terminated, truncated, info)

        if terminated or truncated:
            print(f"!! 超时or完成 超时中止={terminated} 周期完成={truncated}")
            observation, info = env.reset()
            print('!!', observation, info)

    env.close()


if __name__ == '__main__':
    run_random()
