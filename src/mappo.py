"""
MAPPO (Multi-Agent PPO) implementation for stable-baselines3.
This implementation trains multiple PPO agents, one for each agent in the multi-agent environment.
Uses a simpler approach: each agent is trained independently with its own PPO instance.
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class SingleAgentWrapper(gym.Env):
    """
    wraps a multi-agent environment
    """
    
    def __init__(self, multi_agent_env, agent_id: str, max_episode_steps: int = 5000):
        super().__init__()
        self.multi_agent_env = multi_agent_env
        self.agent_id = agent_id
        self.observation_space = multi_agent_env.observation_space
        self.action_space = multi_agent_env.action_space
        self.metadata = multi_agent_env.metadata
        
        self.current_obs = None
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
    
    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self.multi_agent_env.reset(seed=seed, options=options)
        self.current_obs = obs_dict[self.agent_id]
        self.current_step = 0
        info = info_dict.get(self.agent_id, {})
        return self.current_obs, info
    
    def step(self, action):
        actions = {}
        for aid in self.multi_agent_env.agent_ids:
            if aid == self.agent_id:
                actions[aid] = action
            else:
                actions[aid] = 0  # NO_OP for other agents
        
        obs_dict, rew_dict, done_dict, trunc_dict, info_dict = self.multi_agent_env.step(actions)
        
        self.current_obs = obs_dict[self.agent_id]
        reward = rew_dict[self.agent_id]
        done = done_dict[self.agent_id]
        truncated = trunc_dict.get(self.agent_id, False)
        info = info_dict.get(self.agent_id, {})
        self.current_step += 1
        if self.current_step > self.max_episode_steps:
            truncated = True
        return self.current_obs, reward, done, truncated, info
    
    def render(self):
        return self.multi_agent_env.render()
    
    def close(self):
        return self.multi_agent_env.close()


class MAPPO:
    """
    trains separate PPO agents for each agent in the multi-agent environment
    """
    
    def __init__(
        self,
        policy: str,
        multi_agent_env_factory: callable,
        num_agents: int,
        shared_policy: bool = False,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        device: str = "cpu",
        **ppo_kwargs
    ):
        self.num_agents = num_agents
        self.shared_policy = shared_policy
        self.multi_agent_env_factory = multi_agent_env_factory
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        
        if shared_policy:
            # create one shared PPO agent
            def make_shared_env():
                base_env = multi_agent_env_factory()
                return SingleAgentWrapper(base_env, self.agent_ids[0])
            
            vec_env = DummyVecEnv([make_shared_env])
            
            self.agents = [PPO(
                policy,
                vec_env,
                verbose=verbose,
                tensorboard_log=tensorboard_log,
                device=device,
                **ppo_kwargs
            )]
        else:
            # create separate PPO agent for each
            self.agents = []
            for i, agent_id in enumerate(self.agent_ids):
                def make_agent_env(agent_id=agent_id):
                    base_env = multi_agent_env_factory()
                    return SingleAgentWrapper(base_env, agent_id)
                
                vec_env = DummyVecEnv([make_agent_env])
                
                agent = PPO(
                    policy,
                    vec_env,
                    verbose=verbose,
                    tensorboard_log=f"{tensorboard_log}_agent_{i}" if tensorboard_log else None,
                    device=device,
                    **ppo_kwargs
                )
                self.agents.append(agent)
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 1,
        tb_log_name: str = "MAPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        if self.shared_policy:
            # train shared policy
            self.agents[0].learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=log_interval,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
            )
        else:
            # train each agent separately
            timesteps_per_agent = total_timesteps // self.num_agents
            
            for i, agent in enumerate(self.agents):
                print(f"Training agent {i}")
                agent.learn(
                    total_timesteps=timesteps_per_agent,
                    callback=callback,
                    log_interval=log_interval,
                    tb_log_name=f"{tb_log_name}_agent_{i}",
                    reset_num_timesteps=reset_num_timesteps if i == 0 else False,
                    progress_bar=progress_bar,
                )
    
    def predict(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[Dict[str, int], Optional[Dict[str, np.ndarray]]]:
        """Predict actions for all agents"""
        actions = {}
        states = {}
        
        for i, agent_id in enumerate(self.agent_ids):
            if self.shared_policy:
                agent = self.agents[0]
            else:
                agent = self.agents[i]
            
            obs = observation[agent_id]
            obs_reshaped = obs.reshape(1, -1) if len(obs.shape) == 1 else obs
            
            action, state = agent.predict(obs_reshaped, deterministic=deterministic)
            actions[agent_id] = action[0] if isinstance(action, np.ndarray) and len(action.shape) > 0 else action
            states[agent_id] = state
        
        return actions, states
    
    def save(self, path: str):
        """Save agent models"""
        os.makedirs(path, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(path, f"agent_{i}")
            agent.save(agent_path)
    
    @classmethod
    def load(cls, path: str, multi_agent_env_factory: callable, num_agents: int, shared_policy: bool = False):
        """Load model"""
        agents = []
        agent_ids = [f"agent_{i}" for i in range(num_agents)]
        
        if shared_policy:
            agents = [PPO.load(path)]
        else: 
            for i in range(num_agents):
                agent_path = os.path.join(path, f"agent_{i}")
                if os.path.exists(agent_path + ".zip"):
                    def make_load_env(agent_id=agent_ids[i]):
                        base_env = multi_agent_env_factory()
                        return SingleAgentWrapper(base_env, agent_id)
                    
                    vec_env = DummyVecEnv([make_load_env])
                    
                    agent = PPO.load(agent_path, env=vec_env)
                    agents.append(agent)
        
        mappo = cls.__new__(cls)
        mappo.num_agents = num_agents
        mappo.shared_policy = shared_policy
        mappo.multi_agent_env_factory = multi_agent_env_factory
        mappo.agent_ids = agent_ids
        mappo.agents = agents
        return mappo

