from dm_control import suite
from dm_control.viewer import launch

# This loads a simple pendulum environment from the DeepMind Control Suite
#env = suite.load(domain_name="cartpole", task_name="balance")
#launch(env)
import gym
import time
#from real.envs import a1_env
from gym.envs.registration import register
from real.envs.a1_env import A1Real

register(
    id="A1Run-v0",
    entry_point="real.envs.a1_env:A1Real",
    max_episode_steps=1000,
)

env = gym.make("A1Run-v0")
obs = env.reset()

for _ in range(500):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample())
    time.sleep(0.01)
    if done:
        env.reset()

env.close()