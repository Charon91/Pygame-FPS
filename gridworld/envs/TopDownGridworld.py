import tkinter
from PIL import Image
from PIL import ImageTk
from scipy.misc import imresize
import numpy as np
from gym import spaces
import gym



MDP = ['XXXXXXXXXXXXX',
       'X.....X.....X',
       'X.....X.....X',
       'X...........X',
       'X.....X.....X',
       'X.....X.....X',
       'XX.XXXX.....X',
       'X.....XXX.XXX',
       'X.....X.....X',
       'X.....X.....X',
       'X...........X',
       'X.....X.....X',
       'XXXXXXXXXXXXX']

ACTION_DIR = {'N': (-1, 0),
                     'E': ( 0, 1),
                     'S': ( 1, 0),
                     'W': ( 0,-1),}

ACTIONS = {0: ACTION_DIR['N'],
           1: ACTION_DIR['E'],
           2: ACTION_DIR['S'],
           3: ACTION_DIR['W'],}


class TopDownGridworld(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)

        self.nb_rows, self.nb_cols = 13, 13
        self.states = self.nb_rows * self.nb_cols
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(self.nb_rows, self.nb_cols, 3))
        self.start_x, self.start_y = 1, 1
        self.agent_x, self.agent_y = 1, 1
        self.goal_x, self.goal_y = 11, 11


        self.win = tkinter.Toplevel()

        screen_width = self.win.winfo_screenwidth()
        screen_height = self.win.winfo_screenheight()

        self.MDP = None
        self.load_map()

        # calculate position x and y coordinates
        x = screen_width + 100
        y = screen_height + 100
        self.h = self.MDP.shape[0] * 30
        self.w = self.MDP.shape[1] * 30
        self.win.geometry('%sx%s+%s+%s' % (self.w, self.h, x, y))
        self.win.title("ThirdPersonGridworld")

    def render(self, mode='human', close=False):
        screen = imresize(self.generate_observation(),
                          [self.h, self.w, 3],
                          interp='nearest')
        screen = Image.fromarray(screen)

        image = ImageTk.PhotoImage(screen)
        label = tkinter.Label(self.win, image=image)
        label.place(x=0, y=0,
                        width=self.w, height=self.h)

        self.win.update_idletasks()
        self.win.update()

    def generate_observation(self):
        rgb_image = np.zeros((self.nb_rows, self.nb_cols, 3)) + 0.  # self.MDP)
        rgb_image[self.MDP == 1] = [0.5, 0.5, 0.5]
        rgb_image[self.agent_x, self.agent_y] = [1., 1., 1.]
        rgb_image[self.goal_x, self.goal_y] = [1, 0., 0.]
        return rgb_image

    def reset(self):
        s = self.get_state_index(self.start_x, self.start_y)
        self.agent_x, self.agent_y = self.start_x, self.start_y
        screen = self.generate_observation()
        return screen

    def load_map(self):
        self.MDP = np.zeros((self.nb_rows, self.nb_cols))
        for i,_ in enumerate(MDP):
            for j,_ in enumerate(MDP):
                if MDP[i][j] == '.':
                    self.MDP[i][j] = 0
                elif MDP[i][j] == 'X':
                    self.MDP[i][j] = 1

    def get_state_index(self, x, y):
        idx = y + x * self.nb_cols
        return idx

    def _get_next_state(self, a):
        nextX, nextY = self.agent_x + ACTIONS[a][0], self.agent_y + ACTIONS[a][1]
        if 0 <= nextX < self.nb_rows and 0 <= nextY < self.nb_rows:
            if self.MDP[nextX][nextY] != 1:
                return nextX, nextY
        return self.agent_x, self.agent_y

    def is_goal(self, nextX, nextY):
        if nextX == self.goal_x and nextY == self.goal_y:
            return True
        else:
            return False

    def get_reward(self, nextX, nextY):
        if nextX == self.goal_x and nextY == self.goal_y:
            return 1
        else:
            return 0

    def get_state(self, idx):
        x, y = self.get_state_xy(idx)
        self.agent_x, self.agent_y = x, y

        screen = self.generate_observation()

        return screen, x, y

    def get_state_xy(self, idx):
        y = idx % self.nb_cols
        x = int((idx - y) / self.nb_rows)
        return x, y

    def is_wall(self, i, j):
        if self.MDP[i][j] != 1:
            return True
        else:
            return False

    def step(self, a):
        self.agent_x, self.agent_y = self._get_next_state(a)

        done = False
        if self.is_goal(self.agent_x, self.agent_y):
            done = True

        reward = self.get_reward(self.agent_x, self.agent_y)
        nextStateIdx = self.get_state_index(self.agent_x, self.agent_y)

        screen = self.generate_observation()
        return screen, reward, done, nextStateIdx

    def bootstrap_next_state(self, idx, a):
        x, y = self.get_state_xy(idx)
        nextX, nextY = x + ACTIONS[a][0], y + ACTIONS[a][1]
        if 0 <= nextX < self.nb_rows and 0 <= nextY < self.nb_rows:
            if self.MDP[nextX][nextY] != 1:
                return self.get_state_index(nextX,nextY)
        return self.get_state_index(x,y)


if __name__ == '__main__':
    env = TopDownGridworld()

    ep = 0
    step = 0
    tot_rw = 0
    ep_r = 0
    s = env.reset()

    while True:
        s, r, d, _ = env.step(np.random.randint(4))
        step += 1
        ep_r += r
        env.render()
        tot_rw += r
        if d:
            ep += 1
            print("ep {} reward is {} ep steps {}".format(ep, ep_r, step))
            ep_r = 0
            step = 0
            s = env.reset()
