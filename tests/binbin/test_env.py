from binbin.builder import get_basic_env
from common.logging import set_logging_level


def main():
    env = get_basic_env()
    print(f'APP周期最小公倍数={env.app_cycle_lcm}')

    set_logging_level(0)
    env.start()
    while env.now < 1000:
        _, is_timeout, is_env_done = env.run_until_allocation_requested_or_timeout_or_done()

        if is_timeout:
            pass

        if is_env_done:
            print(f"is_env_done=True 当前时刻={env.now}")
            break

        while len(env.allocation_queue) > 0:
            task_pool = env.task_pool
            task = env.allocation_queue.pop()
            env.random_allocate(task)

    print(">> 模拟完毕 <<", env.total_cost)


if __name__ == '__main__':
    main()
