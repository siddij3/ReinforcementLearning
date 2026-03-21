import gymnasium as gym
import time

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset() # obs is a matrix of 8 values, info is an empty dict
# print(obs)
for _ in range(1000):
    action = env.action_space.sample()   # random policy for now
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward, terminated, truncated, info)
    time.sleep(0.5)
    if terminated or truncated:
        obs, info = env.reset()