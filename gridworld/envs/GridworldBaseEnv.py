######################################################################
# Parts of this class and general approach for pygame rendering from:#
#   https://github.com/mlambir/Pygame-FPS                            #
#                                                                    #
######################################################################

import gym
import pygame
from pygame.locals import *
import numpy as np
import time
import os
import copy

#from pygame.tests import camera_test
#from skimage.data import camera

FOUR_ROOMS = [
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 2],
    [2, 4, 0, 4, 4, 4, 7, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 5, 6, 6, 0, 6, 6, 2],
    [2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
]

TEXWIDTH = 46
TEXHEIGHT = 46



def load_image(image, darken, colorKey=None):
    ret = []
    if colorKey is not None:
        image.set_colorkey(colorKey)
    if darken:
        image.set_alpha(127)
    for i in range(image.get_width()):
        s = pygame.Surface((1, image.get_height())).convert()
        s.blit(image, (- i, 0))
        if colorKey is not None:
            s.set_colorkey(colorKey)
        ret.append(s)
    return ret


class Camera(object):
    def __init__(self, x, y, dirx, diry, planex, planey):
        self.x = float(x)
        self.y = float(y)
        self.dirx = float(dirx)
        self.diry = float(diry)
        self.planex = float(planex)
        self.planey = float(planey)

        self._start_x = float(x)
        self._start_y = float(y)
        self._start_dirx = float(dirx)
        self._start_diry = float(diry)
        self._start_planex = float(planex)
        self._start_planey = float(planey)

    def reset(self):
        self.x = self._start_x
        self.y = self._start_y
        self.dirx = self._start_dirx
        self.diry = self._start_diry
        self.planex = self._start_planex
        self.planey = self._start_planey

    def randomdir(self):
        self.dirx = np.random.uniform(-1.0, 1.0)
        self.diry = np.random.uniform(-1.0, 1.0)
        scale = np.sqrt(self.dirx ** 2 + self.diry ** 2)
        self.dirx /= scale
        self.diry /= scale

        self.planex = self.diry * 0.5657
        self.planey = self.dirx * -0.5657


facing_dir_compas = {'E': [.0, 1, .5, .0],
                     'NE': [.7, .7, .4, -.4],
                     'S': [1, .0, .0, -.5],
                     'SE': [.7, -.7, -.4, -.4],
                     'W': [.0, -1, -.5, .0],
                     'SW': [-.7, -.7, -.4, .4],
                     'N': [-1, .0, .0, .5],
                     'NW': [-.7, .7, .4, .4]}


facing_dir_encoded = {0: facing_dir_compas['N'],
                      1: facing_dir_compas['NE'],
                      2: facing_dir_compas['E'],
                      3: facing_dir_compas['SE'],
                      4: facing_dir_compas['S'],
                      5: facing_dir_compas['SW'],
                      6: facing_dir_compas['W'],
                      7: facing_dir_compas['NW']}

action_dir_compas = {'N': (-1, 0),
                     'E': ( 0, 1),
                     'S': ( 1, 0),
                     'W': ( 0,-1)}

actions = {0: action_dir_compas['N'],
           1: action_dir_compas['E'],
           2: action_dir_compas['S'],
           3: action_dir_compas['W']}


class GridworldBase(gym.Env):
    def __init__(self, screen_size=(84, 84)):
        self.worldMap = FOUR_ROOMS

        self.nb_cols = len(self.worldMap[0])
        self.nb_rows = len(self.worldMap[1])

        pygame.init()
        self.window_size = self.h, self.w = screen_size
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Visual MDP")
        self.screen = pygame.display.get_surface()
        self.sprite_positions = [(11.5, 11.5)]
        self.camera = Camera(1.5, 1.5, 0.7, 0.7, .4, -.4)
        self.f = pygame.font.SysFont(pygame.font.get_default_font(), 20)
        self.goal = [11, 11]

        self.textures_dir = os.path.join(os.path.dirname(__file__), "../../textures/")

        self.movement_statistics = np.zeros_like(self.worldMap)
        self.background = None

        self.sprites = [
            load_image(pygame.image.load(os.path.join(self.textures_dir, "barrel.png")).convert(), False,
                       colorKey=(0, 0, 0)),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "pillar.png")).convert(), False,
                       colorKey=(0, 0, 0)),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "greenlight.png")).convert(), False,
                       colorKey=(0, 0, 0)),
        ]

        self.background = None
        self.images = [
            load_image(pygame.image.load(os.path.join(self.textures_dir, "eagle.png")).convert(), False),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "redbrick.png")).convert(), False),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "purplestone.png")).convert(), False),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "greystone.png")).convert(), False),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "bluestone.png")).convert(), False),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "mossy.png")).convert(), False),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "wood.png")).convert(), False),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "colorstone.png")).convert(), False),

            load_image(pygame.image.load(os.path.join(self.textures_dir, "eagle.png")).convert(), True),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "redbrick.png")).convert(), True),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "purplestone.png")).convert(), True),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "greystone.png")).convert(), True),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "bluestone.png")).convert(), True),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "mossy.png")).convert(), True),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "wood.png")).convert(), True),
            load_image(pygame.image.load(os.path.join(self.textures_dir, "colorstone.png")).convert(), True),
        ]

    def step(self, action):
        pass

    def reset(self):
        self.camera.reset()
        # self.camera.randomdir()
        self._update_screen()
        obs = pygame.surfarray.array3d(self.screen)
        obs = np.rot90(obs, 3)
        obs = np.fliplr(obs)
        return obs

    def get_state(self):
        return self._xy_to_state(np.ceil(self.camera.x), np.ceil(self.camera.y))

    def get_xy(self):
        return int(np.ceil(self.camera.x)), int(np.ceil(self.camera.y))

    def render(self, close=True, mode='human'):
        self._update_screen()
        pygame.display.flip()

    def close(self):
        super().close()

    def seed(self, seed=None):
        return super().seed(seed)

    def _update_screen(self):
        if self.background is None:
            self.background = pygame.transform.scale(
                pygame.image.load(os.path.join(self.textures_dir, "background.png")).convert(),
                (self.w, self.h))
        self.screen.blit(self.background, (0, 0))
        zBuffer = []
        for x in range(self.w):
            # Position
            cameraX = float(2 * x / float(self.w) - 1)
            rayPosX = self.camera.x
            rayPosY = self.camera.y
            rayDirX = self.camera.dirx + self.camera.planex * cameraX
            rayDirY = self.camera.diry + self.camera.planey * cameraX
            mapX = int(rayPosX)
            mapY = int(rayPosY)

            # length of ray from one x or y-side to next x or y-side
            deltaDistX = np.sqrt(1 + (rayDirY * rayDirY) / ((rayDirX * rayDirX) + 1e-7))
            if rayDirY == 0:
                rayDirY = 0.00001
            deltaDistY = np.sqrt(1 + (rayDirX * rayDirX) / ((rayDirY * rayDirY) + 1e-7))

            hit = 0  # was there a wall hit?
            side = 0  # was a NS or a EW wall hit?

            # calculate step and initial sideDist
            if rayDirX < 0:
                stepX = - 1
                sideDistX = (rayPosX - mapX) * deltaDistX
            else:
                stepX = 1
                sideDistX = (mapX + 1.0 - rayPosX) * deltaDistX

            if rayDirY < 0:
                stepY = - 1
                sideDistY = (rayPosY - mapY) * deltaDistY
            else:
                stepY = 1
                sideDistY = (mapY + 1.0 - rayPosY) * deltaDistY

            # perform DDA
            while hit == 0:
                # jump to next map square, OR in x - direction, OR in y - direction
                if sideDistX < sideDistY:

                    sideDistX += deltaDistX
                    mapX += stepX
                    side = 0
                else:
                    sideDistY += deltaDistY
                    mapY += stepY
                    side = 1

                # Check if ray has hit a wall
                if self.worldMap[mapX][mapY] > 0:
                    hit = 1

            if side == 0:
                perpWallDist = np.abs((mapX - rayPosX + (1 - stepX) / 2) / rayDirX)
            else:
                perpWallDist = np.abs((mapY - rayPosY + (1 - stepY) / 2) / rayDirY)


            if perpWallDist == 0:
                perpWallDist = 0.000001
            lineHeight = abs(int(self.h / perpWallDist))


            drawStart = - lineHeight / 2 + self.h / 2
            texNum = self.worldMap[mapX][mapY] - 1  

            # calculate value of wallX
            if side == 1:
                wallX = rayPosX + ((mapY - rayPosY + (1 - stepY) / 2) / rayDirY) * rayDirX
            else:
                wallX = rayPosY + ((mapX - rayPosX + (1 - stepX) / 2) / rayDirX) * rayDirY
            wallX -= np.floor(wallX)

            # x coordinate on the texture
            texX = int(wallX * float(TEXWIDTH))
            if side == 0 and rayDirX > 0:
                texX = TEXWIDTH - texX - 1
            if side == 1 and rayDirY < 0:
                texX = TEXWIDTH - texX - 1

            if side == 1:
                texNum += 8
            if lineHeight > 10000:
                lineHeight = 10000
                drawStart = -10000 / 2 + self.h / 2
            self.screen.blit(pygame.transform.scale(self.images[texNum][texX], (1, lineHeight)), (x, drawStart))

            zBuffer.append(perpWallDist)

        # draw sprites
        self.sprite_positions.sort(key=lambda a: np.sqrt((a[0] - self.camera.x) ** 2 + (a[1] - self.camera.y) ** 2))
        for sprite in self.sprite_positions:
            # translate sprite position to relative to camera
            spriteX = sprite[0] - self.camera.x
            spriteY = sprite[1] - self.camera.y

            # required for correct matrix multiplication
            invDet = 1.0 / ((
                                    self.camera.planex * self.camera.diry - self.camera.dirx * self.camera.planey) + 1e-7)

            # this is actually the depth inside the self.screen, that what Z is in 3D
            transformX = invDet * (self.camera.diry * spriteX - self.camera.dirx * spriteY)
            transformY = invDet * (
                    -self.camera.planey * spriteX + self.camera.planex * spriteY)

            spritesurfaceX = int((self.w / 2) * (1 + transformX / (transformY + 1e-7)))

            # calculate height of the sprite on self.screen
            spriteHeight = abs(
                int(self.h / (
                            transformY + 1e-7)))  # using "transformY" instead of the real distance prevents fisheye
            # calculate lowest and highest pixel to fill in current stripe
            drawStartY = -spriteHeight / 2 + self.h / 2

            # calculate width of the sprite
            spriteWidth = abs(int(self.h / (transformY + 1e-7)))
            drawStartX = -spriteWidth / 2 + spritesurfaceX
            drawEndX = spriteWidth / 2 + spritesurfaceX

            if spriteHeight < 1000:
                for stripe in range(int(drawStartX), int(drawEndX)):
                    texX = int((256 * (stripe - (-spriteWidth / 2 + spritesurfaceX)) * TEXWIDTH / spriteWidth) / 256)
                    # the conditions in the if are:
                    # 1) it's in front of camera plane so you don't see things behind you
                    # 2) it's on the self.screen (left)
                    # 3) it's on the self.screen (right)
                    # 4) ZBuffer, with perpendicular distance
                    if transformY > 0 and stripe > 0 and stripe < self.w and transformY < zBuffer[stripe]:
                        self.screen.blit(pygame.transform.scale(self.sprites[2][texX], (1, spriteHeight)),
                                         (stripe, drawStartY))

    def _move(self, x, y=0):
        # move forward if no wall in front of you
        moveX = self.camera.x + self.camera.dirx * x
        if 0 < int(moveX) < 13:
            if self.worldMap[int(moveX)][int(self.camera.y)] == 0: #and \
                    #(self.worldMap[int(np.trunc(moveX + 0.5))][int(self.camera.y+0.5)] == 0 or
                    # self.worldMap[int(np.ceil(moveX - 0.5))][int(self.camera.y-0.5)] == 0):
                self.camera.x += self.camera.dirx * x
        moveY = self.camera.y + self.camera.diry * x
        if 0 < int(moveY) < 13:
            if self.worldMap[int(self.camera.x)][int(moveY)] == 0: #and \
                    #(self.worldMap[int(self.camera.x+0.5)][int(np.trunc(moveY + 0.5))] == 0 or
                    # self.worldMap[int(self.camera.x-0.5)][int(np.ceil(moveY - 0.5))] == 0):
                self.camera.y += self.camera.diry * x

    def _is_north_wall(self):
        moveX = self.camera.x + self.camera.dirx * .5
        if self.camera.dirx < 0:
            if self.worldMap[int(moveX)][int(self.camera.y)] != 0 and \
                    (self.worldMap[int(np.trunc(moveX))][int(self.camera.y)] != 0 or
                     self.worldMap[int(np.ceil(moveX))][int(self.camera.y)] != 0):
                return True
        return False

    def _is_south_wall(self):
        moveX = self.camera.x + self.camera.dirx * .5
        if self.camera.dirx > 0:
            if self.worldMap[int(moveX)][int(self.camera.y)] != 0 and \
                    (self.worldMap[int(np.trunc(moveX))][int(self.camera.y)] != 0 or
                     self.worldMap[int(np.ceil(moveX))][int(self.camera.y)] != 0):
                return True
        return False

    def _is_east_wall(self):
        moveY = self.camera.y + self.camera.diry * .5
        if self.camera.diry > 0:
            if self.worldMap[int(moveY)][int(self.camera.y)] != 0 and \
                    (self.worldMap[int(self.camera.x)][int(np.trunc(moveY))] != 0 or
                     self.worldMap[int(self.camera.x)][int(np.ceil(moveY))] != 0):
                return True
        return False

    def _is_west_wall(self):
        moveY = self.camera.y + self.camera.diry * .5
        if self.camera.diry < 0:
            if self.worldMap[int(moveY)][int(self.camera.y)] != 0 and \
                    (self.worldMap[int(self.camera.x)][int(np.trunc(moveY))] != 0 or
                     self.worldMap[int(self.camera.x)][int(np.ceil(moveY))] != 0):
                return True
        return False

    def _check_walls(self):
        return not (self._is_north_wall() or
                    self._is_east_wall() or
                    self._is_west_wall() or
                    self._is_south_wall())

    def _rotate_world(self, rotation):
        # rotate to the right
        # both camera direction and camera plane must be rotated
        oldDirX = self.camera.dirx
        self.camera.dirx = self.camera.dirx * np.cos(- rotation) - self.camera.diry * np.sin(- rotation)
        self.camera.diry = oldDirX * np.sin(- rotation) + self.camera.diry * np.cos(- rotation)
        oldPlaneX = self.camera.planex
        self.camera.planex = self.camera.planex * np.cos(- rotation) - self.camera.planey * np.sin(- rotation)
        self.camera.planey = oldPlaneX * np.sin(- rotation) + self.camera.planey * np.cos(- rotation)

    def _xy_to_state(self, x, y):
        return int(x) * self.nb_cols + int(y)

    def _state_to_xy(self, s):
        return s // self.nb_cols, s % self.nb_cols

    def play(self):
        t = time.clock()  # time of current frame
        oldTime = 0.  # time of previous frame

        size = w, h = 800, 800
        pygame.init()
        # pixScreen = pygame.surfarray.pixels2d(screen)
        pygame.mouse.set_visible(False)
        clock = pygame.time.Clock()

        f = pygame.font.SysFont(pygame.font.get_default_font(), 20)
        i = 0
        while True:
            clock.tick(60)

            if np.ceil(self.camera.x) >= 11.5 and np.ceil(self.camera.y) >= 11.5:
                self.camera.reset()
                # self.camera.randomdir()

            self.render()
            self.movement_statistics[int(self.camera.x)][int(self.camera.y)] += 1
            # timing for input and FPS counter

            frameTime = float(clock.get_time()) / 1000.0  # frameTime is the time this frame has taken, in seconds
            t = time.clock()
            text = f.render(str(clock.get_fps()), False, (255, 255, 0))
            #self.screen.blit(text, text.get_rect(), text.get_rect())
            # weapon.draw(screen, t)
            pygame.display.flip()

            # speed modifiers
            moveSpeed = frameTime * 5.0  # the constant value is in squares / second
            rotSpeed = frameTime * 3.0  # the constant value is in radians / second

            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        return
                else:
                    pass

            keys = pygame.key.get_pressed()
            if keys[K_UP]:
                # move forward if no wall in front of you
                moveX = self.camera.x + self.camera.dirx * moveSpeed
                if (self.worldMap[int(moveX)][int(self.camera.y)] == 0 and
                        self._check_walls()):
                    self.camera.x += self.camera.dirx * moveSpeed
                moveY = self.camera.y + self.camera.diry * moveSpeed
                if (self.worldMap[int(self.camera.x)][int(moveY)] == 0 and
                        self._check_walls()):
                    self.camera.y += self.camera.diry * moveSpeed
            if keys[K_DOWN]:
                # move backwards if no wall behind you
                if (self.worldMap[int(self.camera.x - self.camera.dirx * moveSpeed)][
                    int(self.camera.y)] == 0): self.camera.x -= self.camera.dirx * moveSpeed
                if (self.worldMap[int(self.camera.x)][
                    int(self.camera.y - self.camera.diry * moveSpeed)] == 0):
                    self.camera.y -= self.camera.diry * moveSpeed
            if (keys[K_RIGHT] and not keys[K_DOWN]) or (keys[K_LEFT] and keys[K_DOWN]):
                # rotate to the right
                # both camera direction and camera plane must be rotated
                oldDirX = self.camera.dirx
                self.camera.dirx = self.camera.dirx * np.cos(- rotSpeed) - self.camera.diry * np.sin(- rotSpeed)
                self.camera.diry = oldDirX * np.sin(- rotSpeed) + self.camera.diry * np.cos(- rotSpeed)
                oldPlaneX = self.camera.planex
                self.camera.planex = self.camera.planex * np.cos(- rotSpeed) - self.camera.planey * np.sin(- rotSpeed)
                self.camera.planey = oldPlaneX * np.sin(- rotSpeed) + self.camera.planey * np.cos(- rotSpeed)
            if (keys[K_LEFT] and not keys[K_DOWN]) or (keys[K_RIGHT] and keys[K_DOWN]):
                # rotate to the left
                # both camera direction and camera plane must be rotated
                oldDirX = self.camera.dirx
                self.camera.dirx = self.camera.dirx * np.cos(rotSpeed) - self.camera.diry * np.sin(rotSpeed)
                self.camera.diry = oldDirX * np.sin(rotSpeed) + self.camera.diry * np.cos(rotSpeed)
                oldPlaneX = self.camera.planex
                self.camera.planex = self.camera.planex * np.cos(rotSpeed) - self.camera.planey * np.sin(rotSpeed)
                self.camera.planey = oldPlaneX * np.sin(rotSpeed) + self.camera.planey * np.cos(rotSpeed)
            if keys[K_s]:
                print('camera.x {0:1.3f} camera.y {1:1.3f} s {2:d} dirx {3:1.3f} diry {4:1.3f} planex {5:1.3f} planey {6:1.3f}'.format(self.camera.x,
                                                                                                self.camera.y,
                                                                                                self._xy_to_state(self.camera.x, self.camera.y),
                                                                                                self.camera.dirx,
                                                                                                self.camera.diry,
                                                                                                self.camera.planex,
                                                                                                self.camera.planey))
                #print('planex {} planey {} = {}'.format(self.camera.planex, self.camera.planey, self.camera.planex + self.camera.planey))
            if keys[K_r]:
                self.camera.reset()
            if keys[K_m]:
                print(self.get_movement_statistics())
            if keys[K_p]:
                obs = pygame.surfarray.array3d(self.screen)
                obs = np.rot90(obs, 3)
                obs = np.fliplr(obs)
                fig = plt.figure(frameon=False)
                fig.set_size_inches(4,4)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.imshow(obs)
                plt.savefig('./screenshot.png')
            if keys[K_o]:
                s = self.get_state()
                s_img = self.get_frame_360(s)
                fig = plt.figure(frameon=False)
                fig.set_size_inches(4,4)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.imshow(s_img)
                plt.savefig('./screenshot_360.png')
            if keys[K_d]:
                oldDirX = self.camera.dirx
                self.camera.dirx = self.camera.dirx * np.cos(-.8) - self.camera.diry * np.sin(-.8)
                self.camera.diry = oldDirX * np.sin(-.8) + self.camera.diry * np.cos(-.8)
                oldPlaneX = self.camera.planex
                self.camera.planex = self.camera.planex * np.cos(-.8) - self.camera.planey * np.sin(-.8)
                self.camera.planey = oldPlaneX * np.sin(-.8) + self.camera.planey * np.cos(-.8)
                print(0.1)
            if keys[K_ESCAPE]:
                return

    def _isWall_xy(self, x, y):
        if 0 <= x < self.nb_rows and 0 <= y < self.nb_cols:
            return self.worldMap[x][y] != 0
        return True

    def _isWall_state(self, s):
        x, y = self._state_to_xy(s)
        return self.worldMap[x][y] > 0

    def get_movement_statistics(self):
        return np.round(self.movement_statistics[:] / self.movement_statistics.max(), 2)

    def setCameraPosition_xy(self, x, y, direction):
        if isinstance(direction, str):
            dir = facing_dir_compas[direction]
        else:
            dir = facing_dir_encoded[direction]
        self.camera.x = x
        self.camera.y = y
        self.camera.dirx = dir[0]
        self.camera.diry = dir[1]
        self.camera.planex = dir[2]
        self.camera.planey = dir[3]

    def setCameraPosition_state(self, s, direction):
        x, y = self._state_to_xy(s)
        self.setCameraPosition_xy(x, y , direction)

    def get_frame(self, s, direction):
        camera_before = copy.deepcopy(self.camera)
        self.setCameraPosition_state(s, direction)
        self._update_screen()
        obs = pygame.surfarray.array3d(self.screen)
        obs = np.rot90(obs, 3)
        obs = np.fliplr(obs)
        self.camera = camera_before
        return obs

    def get_s1_frame(self, s, direction):
        x, y = self._state_to_xy(s)
        a_x, a_y = action_dir_compas[direction]
        x1 = x + a_x
        y1 = y + a_y
        s1 = self._xy_to_state(x1, y1)
        if not self._isWall_xy(x1, y1):
            return self.get_frame(s1, direction), s1
        else:
            return self.get_frame(s, direction), s

    def get_frames_in_direction(self, direction):
        frames = []
        for i in range(self.nb_cols * self.nb_rows):
            if not self._isWall_state(i):
                frames.append(self.get_frame(i, direction))
        return frames

    def get_frames_in_state(self, s):
        frames = []
        for dir in facing_dir_encoded.keys():
            if not self._isWall_state(s):
                frames.append(self.get_frame(s, dir))
                plt.imshow(self.get_frame(s, dir))
                plt.show()
        return frames

    def get_frames_360(self):
        batch = []
        size = self.h
        for i in range(169):
            if not self._isWall_xy(i % 13, i // 13):
                N = self.get_frame(i, 'N')
                E = self.get_frame(i, 'E')
                S = self.get_frame(i, 'S')
                W = self.get_frame(i, 'W')

                s = np.zeros((size * 4, size, 3))
                s[:size] = N
                s[size:size * 2] = W
                s[size * 2:size * 3] = S
                s[size * 3:size * 4] = E

                batch.append(s / 255)
        return np.asarray(batch)

    def get_frame_360(self, state):
        size = self.h
        s = np.zeros((self.h * 4, self.h, 3))
        if not self._isWall_xy(state // 13, state % 13):
            N = self.get_frame(state, 'N')
            E = self.get_frame(state, 'E')
            S = self.get_frame(state, 'S')
            W = self.get_frame(state, 'W')

            s[:size] = N
            s[size:size * 2] = W
            s[size * 2:size * 3] = S
            s[size * 3:size * 4] = E
        return s / 255

    def get_s1_frame_360(self, s, direction):
        x, y = self._state_to_xy(s)
        a_x, a_y = action_dir_compas[direction]
        x1 = x + a_x
        y1 = y + a_y
        s1 = self._xy_to_state(x1, y1)
        if not self._isWall_xy(x1, y1):
            return self.get_frame_360(s1), s1
        else:
            return self.get_frame_360(s), s

    def _rotate_world_90(self):
        oldDirX = self.camera.dirx
        self.camera.dirx = self.camera.dirx * np.cos(-1.6) - self.camera.diry * np.sin(-1.6)
        self.camera.diry = oldDirX * np.sin(-1.6) + self.camera.diry * np.cos(-1.6)
        oldPlaneX = self.camera.planex
        self.camera.planex = self.camera.planex * np.cos(-1.6) - self.camera.planey * np.sin(-1.6)
        self.camera.planey = oldPlaneX * np.sin(-1.6) + self.camera.planey * np.cos(-1.6)
        self._update_screen()

    def _get_frame_360(self):
        front = pygame.surfarray.array3d(self.screen)
        front = np.rot90(front, 3)
        front = np.fliplr(front)

        camera_before = copy.deepcopy(self.camera)
        self._rotate_world_90()
        right = pygame.surfarray.array3d(self.screen)
        right = np.rot90(right, 3)
        right = np.fliplr(right)

        self._rotate_world_90()
        back = pygame.surfarray.array3d(self.screen)
        back = np.rot90(back, 3)
        back = np.fliplr(back)

        self._rotate_world_90()
        left = pygame.surfarray.array3d(self.screen)
        left = np.rot90(left, 3)
        left = np.fliplr(left)

        self.camera = camera_before
        self._update_screen()

        size = self.h
        s = np.zeros((self.h * 4, self.h, 3))
        s[:size] = front
        s[size:size * 2] = right
        s[size * 2:size * 3] = back
        s[size * 3:size * 4] = left

        s /= 255.0
        return s

    # def discrete_step(self, discrete_action):
    #     x,y = self.get_xy()
    #     self.movement_statistics[int(x)][int(y)] += 1
    #     x1, y1 = x + actions[discrete_action][0], y + actions[discrete_action][1]
    #     if not self._isWall_xy(x1, y1):
    #         self.setCameraPosition_xy(x1, y1,'N')
    #         return (x1 * self.nb_rows) + y1
    #     return (x * self.nb_rows) + y

    def _is_goal(self):
        return np.trunc(self.camera.x) >= self.goal[0] and np.trunc(self.camera.y) >= self.goal[1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    size =120
    env = GridworldBase((800, 800))
    env.play()

    #env.play()
