"""
Multi-agent wrapper for stable-baselines3 MAPPO.
Converts the single-agent MetroGameEnv to a multi-agent format where each path is controlled by a separate agent.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from mini_metro_env import MetroGameEnv, MAX_STATIONS, MAX_PATHS, MAX_STATIONS_PER_PATH
from config import num_paths


class MultiAgentMetroWrapper(gym.Env):
    """
    Wraps MetroGameEnv to provide a multi-agent interface for MAPPO.
    Each agent controls one path
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.base_env = MetroGameEnv(render_mode=render_mode)
        self.num_agents = num_paths
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        
        # each agent sees all stations + their own path
        station_obs_size = 1 + 1 + 2 + 1 + 1 + self.base_env.num_shape_types + self.base_env.num_shape_types
        total_station_obs_size = MAX_STATIONS * station_obs_size
        path_obs_size = 1 + MAX_STATIONS_PER_PATH
        # each agent gets all stations + their own path
        agent_obs_size = total_station_obs_size + path_obs_size
        
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, 
            shape=(agent_obs_size,), 
            dtype=np.float32
        )
        
        self.action_space = self.base_env.action_space
        
        self.metadata = self.base_env.metadata
        
    def _get_agent_obs(self, agent_id: str, full_obs: np.ndarray) -> np.ndarray:
        """Extract observation for a specific agent."""
        agent_idx = int(agent_id.split("_")[1])
        
        # station data (shared by all agents)
        station_obs_size = 1 + 1 + 2 + 1 + 1 + self.base_env.num_shape_types + self.base_env.num_shape_types
        total_station_obs_size = MAX_STATIONS * station_obs_size
        station_data = full_obs[:total_station_obs_size]
        
        # this agent's path data
        path_chunk_size = 1 + MAX_STATIONS_PER_PATH
        path_offset = total_station_obs_size + agent_idx * path_chunk_size
        path_data = full_obs[path_offset:path_offset + path_chunk_size]
        
        agent_obs = np.concatenate([station_data, path_data], dtype=np.float32)
        return agent_obs
    
    def _convert_agent_action_to_env_action(self, agent_id: str, agent_action: int) -> int:
        """Convert agent action to base env action (they use the same space)."""
        return agent_action
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and return observations for all agents."""
        obs, info = self.base_env.reset(seed=seed, options=options)
        
        # convert to multi-agent
        obs_dict = {}
        info_dict = {}
        for agent_id in self.agent_ids:
            obs_dict[agent_id] = self._get_agent_obs(agent_id, obs)
            info_dict[agent_id] = {
                "action_mask": self._get_agent_action_mask(agent_id, info.get("action_mask", None))
            }
        
        return obs_dict, info_dict
    
    def _get_agent_action_mask(self, agent_id: str, base_mask: Optional[np.ndarray]) -> np.ndarray:
        """Get action mask for a specific agent."""
        if base_mask is None:
            return np.ones(self.action_space.n, dtype=np.int8)
        # agents use the same action space, so return the same mask
        return base_mask.copy()
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:        
        # convert to multi-agent format
        obs_dict = {}
        rew_dict = {}
        done_dict = {}
        trunc_dict = {}
        info_dict = {}
                
        for agent_id in self.agent_ids:
            action = actions[agent_id]
            obs, reward, terminated, truncated, info = self.base_env.step(action)

            obs_dict[agent_id] = self._get_agent_obs(agent_id, obs)
            rew_dict[agent_id] = reward
            done_dict[agent_id] = terminated
            trunc_dict[agent_id] = truncated
            info_dict[agent_id] = {
                "action_mask": self._get_agent_action_mask(agent_id, info.get("action_mask", None)),
                "score": info.get("score", 0),
                "steps": info.get("steps", 0)
            }
        
        return obs_dict, rew_dict, done_dict, trunc_dict, info_dict
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        return self.base_env.close()
    
    @property
    def unwrapped(self):
        return self.base_env

