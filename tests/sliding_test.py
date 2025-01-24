import time

import gymnasium
import sliding_puzzles

# # For image-based puzzles
# env = sliding_puzzles.make(
#     w=3,
#     variation="image",
#     image_folder="imagenet-1k",
#     image_pool_size=2,
#     render_mode="human",
# )

# For number-based puzzles
# env = sliding_puzzles.make(w=3, variation="onehot", render_mode="state")
env = gymnasium.make("SlidingPuzzles-v0", w=3, variation="onehot", render_mode="human")
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Replace with your agent's action
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward, terminated, truncated, info)
    env.render()
    time.sleep(1)

env.close()
