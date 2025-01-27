import os
import time

import gymnasium as gym
import numpy as np
import sliding_puzzles
from stable_baselines3 import DQN

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PATH_MODEL = os.path.join(script_dir, "..", "..", "models")


class SliderNumber:
    def __init__(self, human_render=False):
        self.human_render = human_render
        if self.human_render:
            self.auto_env = gym.make(
                "SlidingPuzzles-v0", w=3, variation="onehot", render_mode="human"
            )
        else:
            self.auto_env = gym.make("SlidingPuzzles-v0", w=3, variation="onehot")

        self.manual_env = gym.make(
            "SlidingPuzzles-v0", w=3, variation="onehot", render_mode="human"
        )

    def auto_run(self, model_name="sliding_number/dqn_slide_latest"):
        self.loaded_model = DQN.load(
            os.path.join(PATH_MODEL, model_name), env=self.auto_env
        )
        obs, self.initial_auto_info = (
            self.auto_env.reset()
        )  # Gym API returns (observation, info)
        done = False
        self.total_reward = 0
        self.steps = []
        self.initial_state = self.auto_env.unwrapped.state.copy()
        while not done and self.total_reward >= -500:
            action, _states = self.loaded_model.predict(obs)
            obs, reward, done, _, _ = self.auto_env.step(int(action))
            self.total_reward += reward
            if self.human_render:
                self.auto_env.render()
                time.sleep(1)
            self.steps.append(int(action))

    def set_slider(self, state):
        if (
            not isinstance(state, np.ndarray)
            or state.dtype != np.int64
            or state.shape != (3, 3)
        ):
            raise ValueError("State must be a numpy array of int64 with shape (3, 3)")
        # self.manual_env.reset()
        # self.manual_env.unwrapped.state = state
        # self.x = self.manual_env.unwrapped.state

        self.y, _ = self.manual_env.reset(
            options={"state": state}
        )  # Example state values
        self.x = self.manual_env.unwrapped.state.copy()

    def transform_to_grid(self, state):
        # Extract indices where the value is 1
        indices = np.where(state == 1)[0]

        # Reshape or permute indices into the desired output shape
        output = np.array(indices).reshape(3, 3)

        return output
