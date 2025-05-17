from dm_control import suite
from dm_control.viewer import launch

# This loads a simple pendulum environment from the DeepMind Control Suite
env = suite.load(domain_name="cartpole", task_name="balance")
launch(env)