# src/run_mappo_agent.py

import os
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from multi_line_metro_env import MultiLineMetroEnv
from gymnasium import spaces

CHECKPOINT_DIR = os.path.abspath("./checkpoints")

LINES = ["red", "yellow", "blue"]

env_config = {
    "lines": LINES,
    "max_episode_steps": 100,
}

# ---------------------------------------------------------
# CRITICAL: Match the observation space modification from training
# ---------------------------------------------------------
_dummy_env = MultiLineMetroEnv(env_config)
single_obs_space = _dummy_env.observation_space[LINES[0]]
single_act_space = _dummy_env.action_space[LINES[0]]
_dummy_env.close()
del _dummy_env

# Apply the SAME fix as in train_mappo.py
if isinstance(single_obs_space, spaces.Box):
    single_obs_space = spaces.Box(
        low=single_obs_space.low.min(), 
        high=1000.0,  # Must match training configuration
        shape=single_obs_space.shape,
        dtype=single_obs_space.dtype
    )

# ---------------------------------------------------------
# BUILD OLD-API PPO (must match training configuration exactly)
# ---------------------------------------------------------
config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    .environment(env=MultiLineMetroEnv, env_config=env_config)
    .multi_agent(
        policies={
            "shared_policy": (
                None,
                single_obs_space,  # FIXED: Now matches training
                single_act_space,  # FIXED: Now matches training
                {},
            )
        },
        policy_mapping_fn=lambda aid, *args, **kw: "shared_policy",
    )
    .training(
        model={"vf_share_layers": True},  # FIXED: Must match training
    )
)

print(f">>> Restoring checkpoint from: {CHECKPOINT_DIR}")

# Check if using new checkpoint format (with policies/ subfolder)
policies_dir = os.path.join(CHECKPOINT_DIR, "policies", "shared_policy")
if os.path.exists(policies_dir):
    print(f">>> Detected new checkpoint format")
    # For new format, just build and restore from main directory
    algo = config.build_algo()
    algo.restore(CHECKPOINT_DIR)
else:
    # Old format
    print(f">>> Detected old checkpoint format")
    algo = config.build_algo()
    algo.restore(CHECKPOINT_DIR)

print(">>> Checkpoint loaded successfully.\n")

# ---------------------------------------------------------
# Retrieve policy (now works because new API disabled)
# ---------------------------------------------------------
policy = algo.get_policy("shared_policy")
print(">>> Policy loaded. Beginning interactive rollout...\n")

# ---------------------------------------------------------
# Rollout loop with SIMPLE ACTION MASKING
# ---------------------------------------------------------
# IMPORTANT: Create environment with render_mode="human" for visualization
# We need to modify MultiLineMetroEnv to pass render_mode to base_env
env = MultiLineMetroEnv(env_config)

# Manually enable rendering on the base environment
env.base_env.render_mode = "human"
import pygame
pygame.init()
pygame.display.set_caption("Metro RL Agent - Inference")
env.base_env.screen = pygame.display.set_mode((1500, 1000))  # Adjust to match config
env.base_env.clock = pygame.time.Clock()

obs, info = env.reset()

cumulative_reward = 0
episode_rewards = []
invalid_action_count = 0
total_actions = 0

for t in range(500):
    actions = {}
    
    # Get action mask (same for all agents in this shared observation setup)
    first_agent = list(info.keys())[0]
    action_mask = info[first_agent]["action_mask"]
    
    # CRITICAL: To avoid all agents choosing the same action, we need to either:
    # 1. Add noise/randomness, or 
    # 2. Make observations line-specific
    # For now, let's add a simple workaround: each agent explores slightly different actions
    
    agent_actions_taken = []  # Track which actions have been chosen

    # Compute actions per agent WITH MASKING
    for agent_idx, (agent_id, agent_obs) in enumerate(obs.items()):
        total_actions += 1
        
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(np.expand_dims(agent_obs, 0)).float()
        
        # Get action logits from policy model
        with torch.no_grad():
            action_logits, _ = policy.model({"obs": obs_tensor})
            action_logits = action_logits.cpu().numpy()[0]
        
        # Mask invalid actions
        valid_indices = np.where(action_mask == 1)[0]
        if len(valid_indices) == 0:
            action = 0
            invalid_action_count += 1
        else:
            # Get valid actions sorted by logits
            valid_logits = action_logits[valid_indices]
            sorted_idx = np.argsort(valid_logits)[::-1]
            
            # CRITICAL FIX: Choose different action for each agent
            # Skip actions already chosen by previous agents
            action = None
            for idx in sorted_idx:
                candidate = int(valid_indices[idx])
                if candidate not in agent_actions_taken:
                    action = candidate
                    agent_actions_taken.append(candidate)
                    break
            
            # Fallback if all top actions taken
            if action is None:
                action = int(valid_indices[sorted_idx[0]])
        
        actions[agent_id] = action

    # Debug: Print actions to see if agent is doing anything
    if t % 50 == 0:
        valid_action_pct = 100 * (1 - invalid_action_count / max(1, total_actions))
        print(f"\n--- Step {t} ---")
        print(f"Actions: {actions}")
        print(f"Valid actions available: {action_mask.sum()}/{len(action_mask)}")
        
        # Decode what action 40 means
        for agent_id, action_id in actions.items():
            if action_id != 0:
                action_info = env.base_env._action_map.get(action_id, {})
                print(f"  {agent_id} action {action_id}: {action_info}")
        
        print(f"Current stations: {len(env.base_env.mediator.stations)}")
        print(f"Current paths: {len(env.base_env.mediator.paths)}")
        print(f"Valid action rate: {valid_action_pct:.1f}%")

    obs, rewards, terminated, truncated, info = env.step(actions)
    
    # Render the environment
    env.render()
    
    # Track cumulative reward
    step_reward = sum(rewards.values())
    cumulative_reward += step_reward
    episode_rewards.append(step_reward)
    
    # Print progress every 50 steps
    if t % 50 == 0:
        print(f"  Step Reward: {step_reward:.2f}, Cumulative: {cumulative_reward:.2f}")

    if terminated.get("__all__") or truncated.get("__all__"):
        print(f"\n{'='*60}")
        print(f">>> Episode ended at step {t}.")
        print(f">>> Total Cumulative Reward: {cumulative_reward:.2f}")
        print(f">>> Average Step Reward: {cumulative_reward/max(t, 1):.2f}")
        print(f">>> Invalid Actions: {invalid_action_count} / {total_actions} ({100*invalid_action_count/max(1,total_actions):.1f}%)")
        print(f">>> Final Score: {info[first_agent]['score']}")
        print(f"{'='*60}")
        break

env.close()
print(">>> Done.")