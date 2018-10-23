import pygame
import sys
import math
import random
#import .base
import time
import random
from gym import spaces
import gym
from pygame.constants import KEYDOWN, KEYUP, K_F15
from pygame.constants import K_w, K_a, K_s, K_d
import numpy as np

WINWIDTH = 480  # width of the program's window, in pixels
WINHEIGHT = 480  # height in pixels
RADIUS = 2      # radius of the circles
PLAYERS = 1      # number of players
SKIP_PROBABILITY = 0
SPEED_CONSTANT = 2
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
P1COLOUR = RED
P2COLOUR = GREEN
P3COLOUR = BLUE
BG_COLOR = (25, 25, 25)
BEAM_SIGHT = 240

# basically just holds onto all of them
class AchtungPlayer:

    def __init__(self, color, width):
        self.color = color
        self.score = 0
        self.skip = 0
        # generates random position and direction
        self.width = width
        self.x = random.randrange(50, WINWIDTH - WINWIDTH/4)
        self.y = random.randrange(50, WINHEIGHT - WINHEIGHT/4)
        self.angle = random.randrange(0, 360)

        self.sight = BEAM_SIGHT
        self.dleft90 = BEAM_SIGHT
        self.dleft45 = BEAM_SIGHT
        self.dright90 = BEAM_SIGHT
        self.dright45 = BEAM_SIGHT
        self.dforward = BEAM_SIGHT

    def move(self):
        # computes current movement
        self.x += int(RADIUS * SPEED_CONSTANT * math.cos(math.radians(self.angle)))
        self.y += int(RADIUS * SPEED_CONSTANT * math.sin(math.radians(self.angle)))

    def beambounce(self, current_angle, screen):
        _distance = 0

        for i in range(1, self.sight + 1):
            _distance = self.sight
            _x = self.x + i * int(RADIUS * SPEED_CONSTANT * math.cos(math.radians(current_angle)))
            _y = self.y + i * int(RADIUS * SPEED_CONSTANT * math.sin(math.radians(current_angle)))

            x_check = (_x <= 0) or (_x >= WINWIDTH)
            y_check = (_y <= 0) or (_y >= WINHEIGHT)

            if (x_check or y_check):
                d_bounce = True
            else:
                d_bounce = screen.get_at((_x, _y)) != BG_COLOR

            if d_bounce or i == self.sight:
                _distance = math.sqrt((self.x - _x) ** 2 + (self.y - _y) ** 2)
                break

        return _distance

    def beam(self, screen):
        # Sends out beams in 5 directions in order to see the freespace distance

        left90angle = self.angle + 90
        left45angle = self.angle + 45
        right90angle = self.angle - 90
        right45angle = self.angle - 45

        self.dleft90 = self.beambounce(left90angle, screen)
        self.dleft45 = self.beambounce(left45angle, screen)
        self.dright90 = self.beambounce(right90angle, screen)
        self.dright45 = self.beambounce(right45angle, screen)
        self.dforward = self.beambounce(self.angle, screen)

    def draw(self, screen):
        if self.skip:
            self.skip = 0
        elif random.random() < SKIP_PROBABILITY:
            self.skip = 1
        else:
            pygame.draw.circle(screen, self.color, (self.x, self.y), self.width)

    def update(self):
        self.move()


class AchtungDieKurve(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    """
    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    init_length : int (default: 3)
        The starting number of segments the snake has. Do not set below 3 segments. Has issues with hitbox detection with the body for lower values.

    """

    def __init__(self,
                 width=WINWIDTH,
                 height=WINHEIGHT, fps=30, frame_skip=1, num_steps=1,
                 force_fps=True, display_screen=False, add_noop_action=True, rng=24):

        self.actions = {
            "left": K_a,
            "right": K_d,
        }

        self.score = 0.0  # required.
        self.lives = 0  # required. Can be 0 or -1 if not required.
        self.ticks = 0
        self.previous_score = 0
        self.frame_count = 0
        self.fps = fps
        self.frame_skip = frame_skip
        self.num_steps = num_steps
        self.force_fps = force_fps
        self.viewer = None
        self.add_noop_action = add_noop_action
        self.display_screen = display_screen
        self.last_action = []
        self.action = []
        self.height = height
        self.width = width
        self.screen_dim = (width, height)  # width and height
        self.allowed_fps = None  # fps that the game is allowed to run at.
        self.NOOP = K_F15  # the noop key
        self.rng = None
        self._action_set = self.getActions()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=WINWIDTH, shape=(8,), dtype = np.uint8)

        self.rewards = {    # TODO: take as input
                    "positive": 1.0,
                    "negative": -1.0,
                    "tick": 1,
                    "loss": 0,
                    "win": 5.0
                }
        self.BG_COLOR = BG_COLOR

        self._setup()
        self.init()

    def _setup(self):
        """
        Setups up the pygame env, the display and game clock.
        """
        pygame.init()
        self.screen = pygame.display.set_mode(self.getScreenDims(), 0, 32)
        self.clock = pygame.time.Clock()

    def getActions(self):
        """
        Gets the actions the game supports. Optionally inserts the NOOP
        action if PLE has add_noop_action set to True.

        Returns
        --------

        list of pygame.constants
            The agent can simply select the index of the action
            to perform.

        """
        actions = self.actions
        if isinstance(actions, dict):
            actions = actions.values()

        actions = list(actions)

        if self.add_noop_action:
            actions.append(self.NOOP)

        return actions

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if key == self.actions["left"]:
                    self.player.angle -= 10

                if key == self.actions["right"]:
                    self.player.angle += 10

    def setAction(self, action, last_action):
        """
        Pushes the action to the pygame event queue.
        """
        if action is None:
            action = self.NOOP

        if last_action is None:
            last_action = self.NOOP

        kd = pygame.event.Event(KEYDOWN, {"key": action})
        ku = pygame.event.Event(KEYUP, {"key": last_action})

        pygame.event.post(kd)
        pygame.event.post(ku)

    def _setAction(self, action):
        """
            Instructs the game to perform an action if its not a NOOP
        """

        if action is not None:
            self.setAction(action, self.last_action)

        self.last_action = action

    def act(self, action):
        """
        Perform an action on the game. We lockstep frames with actions. If act is not called the game will not run.

        Parameters
        ----------

        action : int
            The index of the action we wish to perform. The index usually corresponds to the index item returned by getActionSet().

        Returns
        -------

        int
            Returns the reward that the agent has accumlated while performing the action.

        """
        return sum(self._oneStepAct(action) for i in range(self.frame_skip))

    def _oneStepAct(self, action):
        """
        Performs an action on the game. Checks if the game is over or if the provided action is valid based on the allowed action set.
        """

        if self.game_over():
            return 0.0

        if action not in self.getActions():
            action = self.NOOP

        self._setAction(action)
        for i in range(self.num_steps):
            time_elapsed = self._tick()
            self.__step(time_elapsed)
            self._draw_frame()

        self.frame_count += self.num_steps

        return self._getReward()

    def _getReward(self):
        """
        Returns the reward the agent has gained as the difference between the last action and the current one.
        """
        reward = self.getScore() - self.previous_score
        self.previous_score = self.getScore()

        return reward

    def getScreenRGB(self):
        """
        Returns the current game screen in RGB format.

        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).

        """

        return pygame.surfarray.array3d(
            pygame.display.get_surface()).astype(np.uint8)

    def getScreenGrayscale(self):
        """
        Gets the current game screen in Grayscale format. Converts from RGB using relative lumiance.

        Returns
        --------
        numpy uint8 array
                Returns a numpy array with the shape (width, height).


        """
        frame = self.getScreenRGB()
        frame = 0.21 * frame[:, :, 0] + 0.72 * \
            frame[:, :, 1] + 0.07 * frame[:, :, 2]
        frame = np.round(frame).astype(np.uint8)

        return frame

    def getFrameNumber(self):
        """
        Gets the current number of frames the agent has seen
        since PLE was initialized.

        Returns
        --------

        int

        """

        return self.frame_count

    def _tick(self):
        """
        Calculates the elapsed time between frames or ticks.
        """
        if self.force_fps:
            return 1000.0 / self.fps
        else:
            return self.tick(self.fps)

    def tick(self, fps):
        """
        This sleeps the game to ensure it runs at the desired fps.
        """
        return self.clock.tick_busy_loop(fps)

    def adjustRewards(self, rewards):
        """

        Adjusts the rewards the game gives the agent

        Parameters
        ----------
        rewards : dict
            A dictonary of reward events to float rewards. Only updates if key matches those specificed in the init function.

        """
        for key in rewards.keys():
            if key in self.rewards:
                self.rewards[key] = rewards[key]

    def getGameState(self):
        """

        Returns
        -------

        dict
            * x position.
            * y position.

            See code for structure.

        """

        state = [self.player.x, self.player.y, self.player.angle, self.player.dleft90, self.player.dleft45, self.player.dright90, self.player.dright45, self.player.dforward]

        return state

    def getScreenDims(self):
        """
        Gets the screen dimensions of the game in tuple form.

        Returns
        -------
        tuple of int
            Returns tuple as follows (width, height).

        """
        return self.screen_dim

    def getScore(self):
        return self.score

    def game_over(self):
        return self.lives == -1

    def _draw_frame(self):
        """
        Decides if the screen will be drawn too
        """

        if self.display_screen:
            pygame.display.update()

    def setRNG(self, rng):
        """
        Sets the rng for games.
        """

        if self.rng is None:
            self.rng = rng

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        self.player = AchtungPlayer(RED, RADIUS)
        self.screen.fill(self.BG_COLOR)
        self.score = 0
        self.ticks = 0
        self.lives = 1
        self.sight = BEAM_SIGHT

    def __step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt /= 1000.0

        self.ticks += 1
        self._handle_player_events()
        self.score += self.rewards["tick"]
        self.player.update()
        collide_check = 0
        try:
            x_check = (self.player.x < 0) or \
                      (self.player.x > self.width)
            y_check = (self.player.y < 0) or \
                      (self.player.y > self.height)

            collide_check = self.screen.get_at((self.player.x, self.player.y)) != self.BG_COLOR
        except IndexError:
            x_check = (self.player.x < 0) or (self.player.x > self.width)
            y_check = (self.player.y < 0) or (self.player.y > self.height)

        if self.player.skip:
            collide_check = 0
        if x_check or y_check or collide_check:
            self.lives = -1

        if self.lives <= 0.0:
            self.score += self.rewards["loss"]

        self.player.beam(self.screen)

        self.player.draw(self.screen)

    def step(self, a):
        reward = self.act(self._action_set[a])
        state = self.getGameState()
        terminal = self.game_over()
        return state, reward, terminal, {}

    def reset(self):
        #self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_dim[0], self.screen_dim[1], 3), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=WINWIDTH, shape=(8,), dtype = np.uint8)
        self.last_action = []
        self.action = []
        self.previous_score = 0.0
        self.init()
        state = self.getGameState()
        return state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.getScreenRGB()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def seed(self, seed):
        rng = np.random.RandomState(seed)
        self.rng = rng
        self.init()


if __name__ == "__main__":

    pygame.init()
    game = AchtungDieKurve(width=WINWIDTH, height=WINHEIGHT)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        if game.game_over():
            game.init()

        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()
