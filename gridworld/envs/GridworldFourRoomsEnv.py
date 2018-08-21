from gridworld.envs.GridworldBaseEnv import GridworldBase, Camera
from gym.spaces import Discrete, Box
import numpy as np
import pygame


facing_dir_compas = {'E': [.0, 1, .5, .0],
                     'NE': [.7, .7, .4, -.4],
                     'S': [1, .0, .0, -.5],
                     'SE': [.7, -.7, -.4, -.4],
                     'W': [.0, -1, -.5, .0],
                     'SW': [-.7, -.7, -.4, .4],
                     'N': [-1, .0, .0, .5],
                     'NW': [-.7, .7, .4, .4]}

action_dir_compas = {'N': (-1, 0),
                     'E': ( 0, 1),
                     'S': ( 1, 0),
                     'W': ( 0,-1)}

actions = {0: action_dir_compas['N'],
           1: action_dir_compas['E'],
           2: action_dir_compas['S'],
           3: action_dir_compas['W']}

action_direction = {0: 'N',
                    1: 'E',
                    2: 'S',
                    3: 'W'}


class GridworldFourRooms(GridworldBase):
    def __init__(self, screen_size=(84, 84)):
        super(GridworldFourRooms, self).__init__(screen_size)
        self.camera = Camera(1.5, 1.5, 1., .0, .0, -.5)
        self.action_space = Discrete(4)
        self.observation_space = Box(0, 1, (84, 84, 3))
        self.wall_distance = 0.3

    def step(self, action):
        x, y = self.get_xy()
        self.movement_statistics[int(x)][int(y)] += 1
        x1, y1 = x + actions[action][0], y + actions[action][1]

        reward, done = 0, False

        if not self._isWall_xy(x1, y1):
            if self._is_goal():
                reward, done = 1, True
                self.reset()
            else:
                if self._is_north_wall():
                    x1 += self.wall_distance
                if self._is_south_wall():
                    x1 -= self.wall_distance
                if self._is_east_wall():
                    y1 += self.wall_distance
                if self._is_west_wall():
                    y1 -= self.wall_distance
                self.setCameraPosition_xy(x1, y1, action_direction[action])

            self._update_screen()

        info = {'x': self.camera.x,
                'y': self.camera.y,
                's': self._xy_to_state(np.ceil(self.camera.x), np.ceil(self.camera.y))}

        self.movement_statistics[int(self.camera.x)][int(self.camera.y)] += 1
        # pygame.display.flip()
        obs = pygame.surfarray.array3d(self.screen)
        obs = np.rot90(obs, 3)
        obs = np.fliplr(obs)
        return obs, reward, done, info


class GridworldFourRooms360(GridworldFourRooms):
    def __init__(self, screen_size=(84, 84)):
        super(GridworldFourRooms360, self).__init__(screen_size)
        self.observation_space = Box(0, 1, (84*4, 84, 3))

    def step(self, action):
        _, reward, done, info = super().step(action)
        return self._get_frame_360(), reward, done, info


if __name__ == '__main__':
    env = GridworldFourRooms()
    steps = 0
    while True:
        a = env.action_space.sample()
        s, _, d, _ = env.step(a)
        steps += 1
        if d:
            print('Done in {}'.format(steps))
            steps = 0
        env.render()