import gym
import pandas as pd


def simulate(environment_name, model, env=None, max_iterations=10**3, render=False):
    if env is None:
        env = gym.make(environment_name)
    observation = env.reset()

    history = [[0, observation, None, 0]] # i, obs, action, interval_reward

    for i in range(max_iterations):
        if render:
            env.render()

        action = model.choose_action(observation)
        observation, reward, done, info = env.step(action)
        history.append([i + 1, observation, action, reward])

        if done:
            break

    return pd.DataFrame(history)
