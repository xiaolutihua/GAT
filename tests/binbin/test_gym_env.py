from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from binbin.gym_env import GymEnv
from common.logging import set_logging_level


def test_env():
    set_logging_level(1)

    eps_lens, acc_rewards = [], []
    terminated_count, done_count = 0, 0

    env = GymEnv()
    observation, info = env.reset()
    print("0000 !!", observation, info)
    for i in range(1, 1000):
        acc_reward = 0.
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            print("<<", f'action={action}', observation, reward, terminated, truncated, info)
            acc_reward = float(reward) + 0.98 * acc_reward

            if terminated or truncated:
                if terminated:
                    terminated_count += 1
                    print("!! TERMINATED")
                if truncated:
                    done_count += 1
                    print("!! TRUNCATED")
                eps_lens.append(info['step_count'])
                acc_rewards.append(acc_reward)

                observation, info = env.reset()
                print(f'{i:04}', '!!', observation, info, terminated, truncated)
                break
    env.close()

    fig = plt.figure(tight_layout=True)
    gs = GridSpec(2, 4)

    ax = fig.add_subplot(gs[0, :3])
    ax.plot(eps_lens)
    ax.set_title("episode length")
    ax = fig.add_subplot(gs[0, 3:])
    ax.boxplot(eps_lens)
    ax.set_title("episode length")

    ax = fig.add_subplot(gs[1, :3])
    ax.plot(acc_rewards)
    ax.set_title("acc reward")
    ax = fig.add_subplot(gs[1, 3:])
    ax.boxplot(acc_rewards)
    ax.set_title("acc reward")

    print(f'terminated_count={terminated_count}, done_count={done_count}')
    plt.show()


if __name__ == '__main__':
    test_env()
