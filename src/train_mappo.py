import os
import shutil
import numpy as np
import torch 
from ray.rllib.algorithms.ppo import PPOConfig
from multi_line_metro_env import MultiLineMetroEnv
from gymnasium import spaces 

LINES = ["red", "yellow", "blue"]
env_config = {"lines": LINES}

_dummy_env = MultiLineMetroEnv(env_config)
single_obs_space = _dummy_env.observation_space[LINES[0]]
single_act_space = _dummy_env.action_space[LINES[0]]
_dummy_env.close()
del _dummy_env


if isinstance(single_obs_space, spaces.Box):
    single_obs_space = spaces.Box(
        low=single_obs_space.low.min(), 
        high=1000.0,
        shape=single_obs_space.shape,
        dtype=single_obs_space.dtype
    )


CKPT_DIR = os.path.abspath("./checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


## mappo config
config = (
    PPOConfig()
    .environment(
        env=MultiLineMetroEnv,
        env_config=env_config
    )
    .debugging(log_level="CRITICAL") 
    .api_stack(                                
        enable_rl_module_and_learner=False, 
        enable_env_runner_and_connector_v2=False
    )
    .multi_agent(
        policies={
            "shared_policy": (
                None,
                single_obs_space, 
                single_act_space,
                {},
            ),
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
    )
    .env_runners(
        num_env_runners=6, 
        rollout_fragment_length=682,
    )
    .training(
        model={"vf_share_layers": True}, 
        train_batch_size=4092, 
        minibatch_size=128,
        num_epochs=5,
        lr=5e-5, 
    )
)


print("Checking for previous checkpoint")

CKPT_FILE = os.path.join(CKPT_DIR, "algorithm_state.pkl")

if os.path.exists(CKPT_FILE):
    print(f"Restoring from base directory: {CKPT_DIR}")
    algo = config.build_algo() 
    algo.restore(CKPT_DIR) 
else:
    print("No previous checkpoint found, training from scratch.")
    algo = config.build_algo() 



print("\n" + "="*80)
print(f"{'Iter':<6} | {'Reward (Mean)':<15} | {'Eps (Iter)':<12} | {'Eps (Total)':<12} | {'Timesteps':<12}")
print("="*80)

start_iter = 1
if os.path.exists(CKPT_FILE):
    start_iter = int(algo.training_iteration) + 1 
    
for i in range(start_iter, 1001):
    result = algo.train()

    metrics = result.get("env_runners", result)
    
    rew_mean = metrics.get("episode_return_mean", np.nan)
    if np.isnan(rew_mean):
        rew_mean = metrics.get("episode_reward_mean", 0.0)

    episodes_this_iter = metrics.get("num_episodes", 0)
    episodes_total = metrics.get("num_episodes_lifetime", 0)
    timesteps_total = result.get("num_env_steps_sampled_lifetime", 0)

    # Print formatted row
    print(f"{i:<6} | {rew_mean:<15.2f} | {episodes_this_iter:<12} | {int(episodes_total):<12} | {int(timesteps_total):<12}")

    if i % 50 == 0:
        algo.save(CKPT_DIR)
        
        print(f"Checkpoint saved directly to: {CKPT_DIR}")

print("="*80)
print("Training Finished.")