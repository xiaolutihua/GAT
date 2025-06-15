
class MazeExpert:
    def __init__(self, size):
        super().__init__()
        self.size = size*2+1
        
    def take_action(self, state):
        x = state[0]
        y = state[1]
        if x>y:
            if x >= self.size:
                return 1
            else:
                return 3
        else:
            if y >= self.size:
                return 3 # 往右
            else:
                return 1
        

