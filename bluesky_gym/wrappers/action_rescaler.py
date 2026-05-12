import numpy as np
from gymnasium import spaces
import gymnasium as gym


class ScaledHeadingAction(gym.ActionWrapper):
    def __init__(self, env, max_deg=90.0):
        super().__init__(env)
        self.max_deg = float(max_deg)
        self.action_space = spaces.Box(-1.0, 1.0, shape=env.action_space.shape, dtype=np.float64)

    def action(self, action):
        return action * self.max_deg
