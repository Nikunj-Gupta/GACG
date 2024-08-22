from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .gather import GatherEnv
from .disperse import DisperseEnv
from .pursuit import PursuitEnv
from .hallway import HallwayEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["gather"] = partial(env_fn, env=GatherEnv)
REGISTRY["disperse"] = partial(env_fn, env=DisperseEnv)
REGISTRY["pursuit"] = partial(env_fn, env=PursuitEnv)
REGISTRY["hallway"] = partial(env_fn, env=HallwayEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII") 


