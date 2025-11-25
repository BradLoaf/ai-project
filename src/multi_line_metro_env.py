import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from mini_metro_env import MetroGameEnv

MAX_EPISODE_STEPS = 1000

class MultiLineMetroEnv(MultiAgentEnv): 
    """
    Multi-agent wrapper for RLlib/MAPPO.
    """
    
    def __init__(self, env_config=None):
        super().__init__()
        if env_config is None: env_config = {}

        self.line_ids = env_config.get("lines", ["line_0", "line_1", "line_2"])
        self.agents = list(self.line_ids)
        self.possible_agents = list(self.line_ids)
        
        self.agent_to_idx = {agent: i for i, agent in enumerate(self.line_ids)}

        self.base_env = MetroGameEnv(render_mode=None)
        
        self.observation_space = spaces.Dict({
            agent: self.base_env.observation_space for agent in self.line_ids
        })
        self.action_space = spaces.Dict({
            agent: self.base_env.action_space for agent in self.line_ids
        })

        self.observation_spaces = self.observation_space.spaces
        self.action_spaces = self.action_space.spaces

        self._episode_step = 0
        self.max_episode_steps = MAX_EPISODE_STEPS

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._episode_step = 0
        self.agents = list(self.possible_agents)
        
        obs_dict = {a: obs for a in self.agents}
        info_dict = {a: info for a in self.agents}
        return obs_dict, info_dict

    def step(self, actions: Dict[str, int]):
        self._episode_step += 1
        
        individual_penalties = {a: 0.0 for a in self.agents}
        
        for agent_id, action_int in actions.items():
            if action_int == 0: continue

            line_idx = self.agent_to_idx[agent_id]
            action_info = self.base_env._action_map.get(action_int)
            
            if action_info:
                success = self.base_env.mediator.apply_action_for_specific_line(
                    line_index=line_idx,
                    action_type=action_info["type"],
                    params=action_info
                )
                if not success:
                    individual_penalties[agent_id] -= 1.0

        global_obs, global_reward, done, truncated, info = self.base_env.step(0)
        
        obs_dict = {}
        reward_dict = {}
        terminated_dict = {}
        truncated_dict = {}
        info_dict = {}
        
        time_limit = self._episode_step >= self.max_episode_steps
        
        for agent_id in self.agents:
            obs_dict[agent_id] = global_obs
            reward_dict[agent_id] = global_reward + individual_penalties[agent_id]
            terminated_dict[agent_id] = done
            truncated_dict[agent_id] = time_limit
            info_dict[agent_id] = info

        terminated_dict["__all__"] = done
        truncated_dict["__all__"] = time_limit

        if done or time_limit:
            self.agents = []

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict
    
    def render(self):
        """Pass through render to base environment"""
        if hasattr(self.base_env, 'render'):
            return self.base_env.render()
        return None

    def close(self):
        """Ensure base environment is properly closed"""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()