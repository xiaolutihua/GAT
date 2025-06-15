from tqdm import tqdm
import numpy as np
import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# 普通的测试使用
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(1):
        with tqdm(total=int(1), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset()
                done = False
                truncated = False
                while not done and not truncated:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-1:])
                                  })
                pbar.update(1)
    return return_list


def train_on_policy_agent_on_mec_env(env, agent, num_episodes, truncate=True):
    return_list = []
    costs = []
    overtimes = []
    over_time_count = 0
    flag_over_time = False
    flag_lose_control = False
    for i in range(1):
        if flag_lose_control:
            break
        with tqdm(total=int(num_episodes), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes)):
                if flag_lose_control:
                    break
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset()
                done = False
                truncated = False
                cost = 0
                while not done and not truncated:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    cost += info["cost"]
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                if done:
                    over_time_count += 1
                else:
                    over_time_count = 0
                if over_time_count > 5 and truncate:
                    flag_lose_control = True
                # if done:
                #     print("环境超时")
                # print(f"本轮的cost值为{cost}")
                return_list.append(episode_return)
                overtimes.append(done)
                agent.update(transition_dict)
                costs.append(cost)
                pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-1:]),
                                  'cost': '%.3f' % cost,
                                  'overtime': done
                                  })
                pbar.update(1)
    return return_list, costs, overtimes


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncate, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)

                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
