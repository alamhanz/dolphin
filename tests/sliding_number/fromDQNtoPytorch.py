import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN

# Load the stable_baselines3 DQN model
model = DQN.load("src/dolphin/models/sliding_number/dqn_slide_latest.zip")


# Define the PyTorch model
class DQNPyTorch(nn.Module):
    def __init__(self, input_dim, output_dim):
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


# Get the dimensions of the input and output layers
input_dim = model.policy.observation_space.shape[0]
output_dim = model.policy.action_space.n
print(input_dim, output_dim)

# Initialize the PyTorch model
pytorch_model = DQNPyTorch(input_dim, output_dim)

# Copy the weights from the stable_baselines3 model to the PyTorch model
pytorch_model.fc1.weight.data = model.policy.q_net.q_net[0].weight.data.clone().detach()
pytorch_model.fc1.bias.data = model.policy.q_net.q_net[0].bias.data.clone().detach()
pytorch_model.fc2.weight.data = model.policy.q_net.q_net[2].weight.data.clone().detach()
pytorch_model.fc2.bias.data = model.policy.q_net.q_net[2].bias.data.clone().detach()
pytorch_model.fc3.weight.data = model.policy.q_net.q_net[4].weight.data.clone().detach()
pytorch_model.fc3.bias.data = model.policy.q_net.q_net[4].bias.data.clone().detach()
pytorch_model.fc4.weight.data = model.policy.q_net.q_net[6].weight.data.clone().detach()
pytorch_model.fc4.bias.data = model.policy.q_net.q_net[6].bias.data.clone().detach()
pytorch_model.fc5.weight.data = model.policy.q_net.q_net[8].weight.data.clone().detach()
pytorch_model.fc5.bias.data = model.policy.q_net.q_net[8].bias.data.clone().detach()

# Save the PyTorch model
torch.save(pytorch_model.state_dict(), "path_to_save_pytorch_model.pth")

# Load the PyTorch model
loaded_model = DQNPyTorch(input_dim, output_dim)
loaded_model.load_state_dict(torch.load("path_to_save_pytorch_model.pth"))
loaded_model.eval()


curr_obs = [
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
]
# # Example input for inference
example_input = torch.tensor(
    curr_obs,
    dtype=torch.float32,
)  # Adjust the input size as needed

# # Perform inference
with torch.no_grad():
    output = loaded_model(example_input)
    max_index = torch.argmax(output).item()

print("Inference output:", output)
print("Index of max output:", max_index)
print(model.predict(curr_obs))
