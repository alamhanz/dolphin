import time

import gymnasium as gym
import sliding_puzzles
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

PATH_MODEL = "models/sliding_number/"


def main():
    # Step 1: Create the environment
    # env = gym.make("CartPole-v1")
    env = gym.make("SlidingPuzzles-v0", w=3, variation="onehot")

    # # Step 2: Initialize the DQN agent
    # model = DQN("MlpPolicy", env, verbose=1)

    # Step 3: Train the model
    # model = dqn = DQN(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=1e-4,
    #     buffer_size=1500000,
    #     batch_size=64,
    #     gamma=0.97,
    #     target_update_interval=10000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.05,
    #     train_freq=4,
    #     learning_starts=10000,
    #     policy_kwargs=dict(net_arch=[256, 256, 128, 64]),
    #     tensorboard_log="./dqn_logs/",
    #     verbose=1,
    # )

    # eval_callback = EvalCallback(
    #     env,
    #     best_model_save_path="./logs/",
    #     log_path="./logs/",
    #     eval_freq=500,
    #     deterministic=True,
    #     render=False,
    # )

    # # Step 4: Save the trained model (optional)
    # model.learn(total_timesteps=1200000)
    # model.save(PATH_MODEL + "dqn_slide_test3")

    model_check = DQN.load(PATH_MODEL + "dqn_slide_test3", env=env)

    # Step 5: Test the trained model
    # obs, info = env.reset()  # Gym API returns (observation, info)
    # done = False
    # total_reward = 0
    # print(obs)

    # while not done and total_reward >= -7500:
    #     action, _states = model_check.predict(obs)
    #     obs, reward, done, _, _ = env.step(
    #         int(action)
    #     )  # Gym step returns (observation, reward, done, info)
    #     total_reward += reward

    # print(f"Total reward: {total_reward}")

    # Step 6: Render the environment to watch the agent (optional)
    # env_test = gym.make("CartPole-v1", render_mode="human")
    env_test = gym.make(
        "SlidingPuzzles-v0", w=3, variation="onehot", render_mode="human"
    )
    obs, info = env_test.reset()  # Gym API returns (observation, info)
    done = False
    total_reward = 0

    while not done and total_reward >= -7500:
        action, _states = model_check.predict(obs)
        obs, reward, done, _, _ = env_test.step(
            int(action)
        )  # Gym step returns (observation, reward, done, info)
        total_reward += reward
        print(reward)
        env_test.render()
        time.sleep(1)

    print(f"Total reward: {total_reward}")
    time.sleep(3)


if __name__ == "__main__":
    main()
