import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .core import Snake


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    def __init__(self, render_mode=None, **kwargs) -> None:
        self.render_mode = render_mode
        self.snake = Snake(**kwargs)

        # Action: 4 hướng tuyệt đối (0..3)
        self.action_space = spaces.Discrete(4)

        # Observation: Snake.observation() trả về 16-d vector
        low = np.array([
            0, 0, 0,       # danger_{straight,right,left}
            0, 0, 0, 0,    # dir_l, dir_r, dir_u, dir_d
            0, 0, 0, 0,    # food_left, right, up, down
            -1.0, -1.0,    # dx, dy (normalized, có thể âm)
            0.0,           # snake length normalized
            0.0, 0.0       # head_norm_x, head_norm_y
        ], dtype=np.float32)

        high = np.array([
            1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1.0, 1.0,
            1.0,
            1.0, 1.0
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.snake.init()
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def step(self, action):
        obs, reward, dead, truncated = self.snake.step(action)
        terminated = dead
        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        return self.snake.observation()

    def _get_info(self):
        return self.snake.info()

    def _render_frame(self):
        self.snake.render()

    def close(self):
        self.snake.close()

    def play(self):
        self.snake.play()
