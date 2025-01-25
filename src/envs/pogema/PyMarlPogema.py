import numpy as np

from pogema import GridConfig
from pogema.envs import _make_pogema


class PyMarlPogema:

    def __init__(self, 
            seed=1, 
            size=10, 
            density=0, 
            num_agents=8, 
            obs_radius=7, 
            max_episode_steps=256, 
            mh_distance=False, 
            map_name=None
        ):
        gc = GridConfig(
            num_agents=num_agents,  # number of agents 
            size=size,              # size of the grid
            density=density,        # obstacle density 
            seed=seed,              # set to None for random obstacles, agents and targets, positions at each reset 
            obs_radius=obs_radius,  # defines field of view 
            max_episode_steps=max_episode_steps, # horizon 
            integration='PyMARL' 
        )
        self._grid_config: GridConfig = gc 

        self.env = _make_pogema(self._grid_config)
        self._mh_distance = mh_distance
        self._observations, _ = self.env.reset()
        self.max_episode_steps = gc.max_episode_steps
        self.episode_limit = gc.max_episode_steps
        self.n_agents = self.env.get_num_agents()

        self.spec = None

    @property
    def unwrapped(self):
        return self

    def step(self, actions):
        self._observations, rewards, terminated, truncated, infos = self.env.step(actions)
        info = {}
        done = all(terminated) or all(truncated)
        if done:
            for key, value in infos[0]['metrics'].items():
                info[key] = value
            
        return sum(rewards), done, info

    def get_obs(self):
        return np.array([self.get_obs_agent(agent_id) for agent_id in range(self.n_agents)])

    def get_obs_agent(self, agent_id):
        return np.array(self._observations[agent_id]).flatten()

    def get_obs_size(self):
        return len(np.array(self._observations[0]).flatten())

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        return len(self.get_state())

    def get_avail_actions(self):
        actions = []
        for i in range(self.env.get_num_agents()):
            actions.append(self.get_avail_agent_actions(i))
        return actions

    # noinspection PyUnusedLocal
    @staticmethod
    def get_avail_agent_actions(agent_id):
        return list(range(5))

    @staticmethod
    def get_total_actions():
        return 5

    def reset(self):
        self._grid_config = self.env.grid_config
        self._observations, _ = self.env.reset()
        return np.array(self._observations).flatten()

    def save_replay(self):
        return

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    }
        return env_info

    @staticmethod
    def get_stats():
        return {}

    def close(self):
        return

    def sample_actions(self):
        return self.env.sample_actions()