import gym
from stable_baselines3 import DQN


def main():
    # Step 1: Create the environment
    env = gym.make("CartPole-v1")

    # Step 2: Initialize the DQN agent
    model = DQN("MlpPolicy", env, verbose=1)

    # Step 3: Train the model
    model.learn(total_timesteps=10000)

    # Step 4: Save the trained model (optional)
    model.save("dqn_cartpole")

    # Step 5: Test the trained model
    obs, info = env.reset()  # Gym API returns (observation, info)
    done = False
    total_reward = 0
    print(obs)

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(
            action
        )  # Gym step returns (observation, reward, done, info)
        print(obs)
        total_reward += reward

    print(f"Total reward: {total_reward}")

    # Step 6: Render the environment to watch the agent (optional)
    env_test = gym.make("CartPole-v1", render_mode="human")
    obs, info = env_test.reset()  # Gym API returns (observation, info)
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env_test.step(
            action
        )  # Gym step returns (observation, reward, done, info)
        env_test.render()


if __name__ == "__main__":
    main()
