import gymnasium as gym

class MyMaze(gym.Env):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size*2+1
        print("current Maze size is {} ".format(self.size))
        self.max_step = size*2 * 2 * 2
        
        self.cur_step = 0
        self.x = 1
        self.y = 1
    
    def reset(self):
        self.x = 1
        self.y = 1
        self.cur_step = 0
        return ([self.x,self.y], {})
    
    def step(self, action: int):
        if self.cur_step > self.max_step:
            return [self.x, self.y], -10000, True, False, {}
    
        self.cur_step += 1
        if action == 0:
            self.y = self.y-1 if self.y > 1 else 1
        elif action == 1: # 往下
            self.y = self.y+1 if self.y <= self.size else self.size
        elif action == 2:
            self.x = self.x-1 if self.x > 1 else 1
        elif action == 3: # 往右
            self.x = self.x+1 if self.x <= self.size else self.size
        reward = - (abs(self.x- self.y)+1)
        # reward = -1
        if (self.x == self.size and self.y == self.size):
            done = True
        else:
            done = False
        return [self.x, self.y], reward, done, False, {}
    
