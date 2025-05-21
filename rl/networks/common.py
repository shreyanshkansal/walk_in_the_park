import flax.linen as nn
from flax.core import FrozenDict
from jax import tree_util

default_init = nn.initializers.xavier_uniform


def soft_target_update(critic_params: FrozenDict,
                       target_critic_params: FrozenDict,
                       tau: float) -> FrozenDict:
    new_target_params = tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau),
                                     critic_params, target_critic_params)

    return new_target_params
