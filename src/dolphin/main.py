import os
import time

import gymnasium as gym
import sliding_puzzles
from stable_baselines3 import DQN

PATH_MODEL = ""


class SliderNumber:
    def __init__(self, human_render=False):
        if human_render:
            self.auto_env = gym.make(
                "SlidingPuzzles-v0", w=3, variation="onehot", render_mode="human"
            )
        else:
            self.auto_env = gym.make("SlidingPuzzles-v0", w=3, variation="onehot")

        self.manual_env = gym.make("SlidingPuzzles-v0", w=3, variation="onehot")

    def auto_run(self, model_name="dqn_slide_test3"):
        self.loaded_model = DQN.load(
            os.path.join(PATH_MODEL, model_name), env=self.auto_env
        )
        self.initial_state, info = (
            self.auto_env.reset()
        )  # Gym API returns (observation, info)
        done = False
        self.total_reward = 0
        self.steps = []
        obs = self.initial_state
        while not done and total_reward >= -500:
            action, _states = self.loaded_model.predict(obs)
            obs, reward, done, _, _ = self.auto_env.step(int(action))
            self.total_reward += reward
            if self.human_render:
                self.auto_env.render()
            self.steps.append(action)
