import os
import time

import gymnasium as gym
import sliding_puzzles
import torch
import torch.nn as nn
from jinja2 import Template

# from stable_baselines3 import DQN

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PATH_MODEL = os.path.join(script_dir, "models")
PATH_TEMPLATE = os.path.join(script_dir, "templates")


# Define the PyTorch model
class DQNPyTorch(nn.Module):
    def __init__(self, input_dim=81, output_dim=4):
        super(DQNPyTorch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class SliderNumber:
    def __init__(self, human_render=False):
        self.human_render = human_render
        if self.human_render:
            self.auto_env = gym.make(
                "SlidingPuzzles-v0", w=3, variation="onehot", render_mode="human"
            )
        else:
            self.auto_env = gym.make("SlidingPuzzles-v0", w=3, variation="onehot")

    def auto_run(self, model_name="sliding_number/dqn_torch_latest.pth"):
        self.loaded_model = DQNPyTorch()
        self.loaded_model.load_state_dict(
            torch.load(os.path.join(PATH_MODEL, model_name))
        )
        self.loaded_model.eval()

        self.initial_state_onehot, self.initial_auto_info = (
            self.auto_env.reset()
        )  # Gym API returns (observation, info)
        done = False
        self.total_reward = 0
        self.steps = []
        self.initial_state = self.auto_env.unwrapped.state.copy()
        obs = self.initial_state_onehot.copy()
        while not done and self.total_reward >= -500:
            # action, _states = self.loaded_model.predict(obs)

            # # Example input for inference
            example_input = torch.tensor(
                obs,
                dtype=torch.float32,
            )  # Adjust the input size as needed

            # # Perform inference
            with torch.no_grad():
                output = self.loaded_model(example_input)
                action = torch.argmax(output).item()

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
