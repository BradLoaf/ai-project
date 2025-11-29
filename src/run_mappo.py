"""
MAPPO evaluation script.
Loads and runs trained MAPPO agents with a UI.
"""
import os
import time
import argparse
import gymnasium as gym
from multi_agent_wrapper import MultiAgentMetroWrapper
from mappo import MAPPO
from config import num_paths


NUM_AGENTS = num_paths
SHARED_POLICY = False


def create_multi_agent_env():
    """Helper function to create the multi-agent environment for evaluation."""
    env = MultiAgentMetroWrapper(render_mode="human")
    return env


def run_mappo(model_folder, n_episodes = 100):
    """
    Loads and runs trained MAPPO agents with a UI.
    """
    if not os.path.exists(model_folder):
        print(f"Error: Model folder '{model_folder}' not found. Aborting.")
        return
    
    print(f"Loading MAPPO model from {model_folder}...")
    
    # load MAPPO model
    model = MAPPO.load(
        model_folder,
        multi_agent_env_factory=create_multi_agent_env,
        num_agents=NUM_AGENTS,
        shared_policy=SHARED_POLICY
    )
    
    print("Model loaded successfully.")
    
    eval_env = create_multi_agent_env()
    episode_rewards = []
    print("Starting evaluation")
    
    for ep in range(n_episodes):
        obs, info = eval_env.reset()

        dones = {agent: False for agent in eval_env.agent_ids}
        truncs = {agent: False for agent in eval_env.agent_ids}
        ep_reward = 0

        while not all(dones[agent] or truncs[agent] for agent in eval_env.agent_ids):
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, truncs, infos = eval_env.step(actions)
            
            ep_reward += sum(rewards.values()) / len(rewards)
        
        episode_rewards.append(ep_reward)

        print("Resetting environment...")
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained MAPPO agents with UI.")
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to the directory containing the saved MAPPO model (agent_0.zip, agent_1.zip, etc.)."
    )
    
    args = parser.parse_args()
    run_mappo(args.model_folder)
