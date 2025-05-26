#! /usr/bin/env python
import os
import numpy as np
import gym
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags

from rl.agents import SACLearner
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym
from env_utils import make_mujoco_env

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'A1Run-v0', 'Environment name.')
flags.DEFINE_integer('eval_episodes', 5, 'Episodes to evaluate.')
flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
flags.DEFINE_string('chkpt_dir', 'saved/checkpoints', 'Checkpoint directory.')
flags.DEFINE_boolean('wandb', True, 'Log to Weights and Biases.')

config_flags.DEFINE_config_file(
    'config',
    'configs/sac_config.py',
    'Path to training config.',
    lock_config=False
)

def main(_):
    if FLAGS.wandb:
        wandb.init(project='a1-eval')
        wandb.config.update(FLAGS)

    # --- Env + Eval Env ---
    if FLAGS.real_robot:
        from real.envs.a1_env import A1Real
        env = A1Real(zero_action=np.asarray([0.05, 0.9, -1.8] * 4))
        eval_env = env  # no separate eval_env for real robot
    else:
        env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history)
        eval_env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history)

    env = wrap_gym(env, rescale_actions=True)
    eval_env = wrap_gym(eval_env, rescale_actions=True)

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env, deque_size=1)

    env.seed(FLAGS.seed)
    eval_env.seed(FLAGS.seed + 42)

    # --- Agent ---
    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space, env.action_space, **kwargs)

    latest_chkpt = checkpoints.latest_checkpoint(FLAGS.chkpt_dir)
    if latest_chkpt is None:
        raise ValueError(f"No checkpoint found in {FLAGS.chkpt_dir}")
    print(f"✅ Restoring agent from {latest_chkpt}")
    agent = checkpoints.restore_checkpoint(latest_chkpt, agent)

    # --- Evaluate ---
    eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)

    # --- Log ---
    for k, v in eval_info.items():
        print(f"Eval {k}: {v}")
        if FLAGS.wandb:
            wandb.log({f"evaluation/{k}": v})

if __name__ == '__main__':
    app.run(main)
