from functools import partial
from smacv2.env import MultiAgentEnv, StarCraft2Env, StarCraftCapabilityEnvWrapper
from .gather import GatherEnv
from .disperse import DisperseEnv
from .pursuit import PursuitEnv
from .hallway import HallwayEnv
from .pogema import PyMarlPogema
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2wrapped"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["gather"] = partial(env_fn, env=GatherEnv)
REGISTRY["disperse"] = partial(env_fn, env=DisperseEnv)
REGISTRY["pursuit"] = partial(env_fn, env=PursuitEnv)
REGISTRY["hallway"] = partial(env_fn, env=HallwayEnv)
REGISTRY["pogema"] = partial(env_fn, env=PyMarlPogema) 

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII") 


