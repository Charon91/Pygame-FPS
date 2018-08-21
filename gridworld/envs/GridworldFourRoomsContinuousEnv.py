from gridworld.envs.GridworldBaseEnv import GridworldBase, Camera
import gym
from gym.spaces import Discrete, Box
import pygame
from pygame.locals import *
import numpy as np
import time
import os
import copy

MAX_ROTATION_ANGLE = 20


class GridworldFourRoomsContinuous(GridworldBase):
    def __init__(self, screen_size=(84, 84)):
        super(GridworldFourRoomsContinuous, self).__init__(screen_size)
        self.camera = Camera(1.5, 1.5, 0.7, 0.7, .4, -.4)
        self.action_space = Box(-1, 1, (2,))
        self.observation_space = Box(0, 255, (84, 84, 3))

    def reset(self):
        return super().reset()

    def step(self, action):
        velocity = np.clip(action[0] + 1, 0.00001, 1)
        rotation = np.clip(action[1], -1, 1)
        radian = np.radians(rotation * MAX_ROTATION_ANGLE)
        self._rotate_world(radian)
        self._move((0.1 / velocity), velocity)
        reward, done = 0, False
        if self._is_goal():
            reward, done = 2, True
            self.reset()

        reward = 22 / (22 - self.camera.x + self.camera.y)
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


class GridworldFourRoomsContinuous360(GridworldFourRoomsContinuous):
    def __init__(self, screen_size=(84, 84)):
        super(GridworldFourRoomsContinuous360, self).__init__(screen_size)
        self.action_space = Box(-1, 1, (2,))
        self.observation_space = Box(0, 255, (84*4, 84, 3))

    def reset(self):
        super().reset()
        return self._get_frame_360()

    def step(self, action):
        _, reward, done, info = super().step(action)
        return self._get_frame_360(), reward, done, info


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = GridworldFourRoomsContinuous()
    steps = 0
    while True:
        a = env.action_space.sample()
        obs, _, d, _ = env.step(a)
        steps +=1
        if d:
            print('Done in {}'.format(steps))
            steps = 0
        env.render()