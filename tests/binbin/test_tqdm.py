from tqdm import tqdm
import time

for _ in tqdm(range(10)):
    # total参数设置进度条的总长度
    pbar = tqdm(total=100)
    for i in range(100):
        time.sleep(0.05)
        # 每次更新进度条的长度
        pbar.update(1)
    # 别忘了关闭占用的资源
    pbar.close()
