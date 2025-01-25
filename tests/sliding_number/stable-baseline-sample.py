import gym
from stable_baselines3 import PPO

env_train = gym.make("MountainCar-v0")

# Reset environment and check the observation
observation, info = env_train.reset()
print(f"Initial observation: {observation}")
print(f"Observation shape: {len(observation)}")

# Initialize PPO model
model = PPO("MlpPolicy", env_train, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_mountaincar")


env = gym.make("MountainCar-v0", render_mode="human")
# Test the model
observation, info = env.reset()
for _ in range(1000):
    action, _state = model.predict(observation, deterministic=True)
    env.render()
    observation, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

env.close()
