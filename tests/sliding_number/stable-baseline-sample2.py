import gym
from stable_baselines3 import DQN

# Create the environment
env_train = gym.make("CartPole-v1")

# Initialize the DQN model
model = DQN(
    "MlpPolicy",  # Multi-layer perceptron policy
    env_train,  # The environment
    learning_rate=1e-3,  # Learning rate
    buffer_size=100000,  # Replay buffer size
    learning_starts=1000,  # Steps before training begins
    batch_size=64,  # Batch size for training
    gamma=0.99,  # Discount factor
    target_update_interval=500,  # Update target network every 500 steps
    train_freq=4,  # Training frequency
    verbose=1,  # Verbosity level (1 = progress printed)
)

# Train the model
model.learn(total_timesteps=20000)

# Save the model
model.save("dqn_cartpole")

env = gym.make("CartPole-v1", render_mode="human")
# Load the trained model
model = DQN.load("dqn_cartpole", env=env)

# Test the model
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, rewards, dones, _, _ = env.step(action)
    env.render()
