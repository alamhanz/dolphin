import os
import time

import gymnasium as gym
import sliding_puzzles
from jinja2 import Template
from stable_baselines3 import DQN

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PATH_MODEL = os.path.join(script_dir, "models")
PATH_TEMPLATE = os.path.join(script_dir, "templates")


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
        self.initial_state_onehot, self.initial_auto_info = (
            self.auto_env.reset()
        )  # Gym API returns (observation, info)
        done = False
        self.total_reward = 0
        self.steps = []
        self.initial_state = self.auto_env.unwrapped.state.copy()
        obs = self.initial_state_onehot.copy()
        while not done and self.total_reward >= -500:
            action, _states = self.loaded_model.predict(obs)
            obs, reward, done, _, _ = self.auto_env.step(int(action))
            self.total_reward += reward
            if self.human_render:
                self.auto_env.render()
                time.sleep(1)
            self.steps.append(int(action))

    ## sliding_puzzles is not allowing to set the state directly
    def get_html_template(self):
        auto_slider_path = os.path.join(
            PATH_TEMPLATE, "sliding_number", "auto_slider.html"
        )
        with open(auto_slider_path, "r") as file:
            auto_slider_template = Template(file.read())
        self.auto_slider = auto_slider_template

        human_slider_path = os.path.join(
            PATH_TEMPLATE, "sliding_number", "human_slider.html"
        )
        with open(human_slider_path, "r") as file:
            human_slider_template = Template(file.read())
        self.human_slider = human_slider_template
